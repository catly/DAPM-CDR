import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from dgllife.model.gnn import GCN
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()



class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype


    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND  # 交换维度
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]),
              tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

def load_clip_to_cpu():
    backbone_name = 'RN101'  # 'RN101'
    url = clip._MODELS[backbone_name]

    model_path = clip._download(url, "./assets")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class PromptLearner(nn.Module):
    def __init__(self, clip_model):
        classnames = ['sensitive', 'resistant']
        super().__init__()
        n_cls = len(classnames)  # 类的个数
        n_ctx = 16 # 16

        dtype = clip_model.dtype  # floadt32
        ctx_dim = clip_model.ln_final.weight.shape[0]  # 获取CLIP模型的最后一层的权重矩阵的形状，然后取第一个维度的值，也就是模型的嵌入维度。模型的输出特征的大小
        clip_imsize = clip_model.visual.input_resolution # 这句话是获取CLIP模型的视觉部分的输入分辨率，也就是模型可以处理的图像的大小
        domainnames = ["LUAD"] + ["SCLC"]
        domainnames = [
            ", a {} cancer.".format(domain) for domain in domainnames
        ]   # prompt Domain-specific 域特殊prompt
        n_dm = 2  # number of domains  # 领域的个数
        n_dmx = 2  # number of domain context
        n = n_dmx + n_ctx
        self.n_dm = n_dm
        self.n_dmx = n_dmx


        naive_prompt_prefix = "a response of a".replace("_", " ")

        ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype) # 创建一个填充了未初始化数据的张量

        nn.init.normal_(ctx_vectors, std=0.02) # 按照正态分布进行初始化
        print("ctx vectors size: ".format(ctx_vectors.size()))
        prompt_prefix = " ".join(["X"] * n)

        domain_vectors = torch.empty(n_dm, n_dmx, ctx_dim, dtype=dtype)  # 创建一个空的向量
        nn.init.normal_(domain_vectors, std=0.02)
        self.domain_vectors = nn.Parameter(domain_vectors) # 让这个张量自动加入到模型的参数列表中，方便进行优化和更新

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        print(f"Number of domain context words (tokens): {n_dmx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        naive_prompts = [
            naive_prompt_prefix + " " + name + "." for name in classnames
        ]

        prompts = [
            prompt_prefix + " " + name + " " + domain + "."
            for domain in domainnames for name in classnames
        ]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # 用来调用CLIP模型的分词函数，这个函数可以将一个字符串转换为一个整数张量，表示该字符串中的每个单词或者子词的编号。
        naive_tokenized_prompts = torch.cat(
            [clip.tokenize(p) for p in naive_prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(
                dtype)
            naive_embedding = clip_model.token_embedding(
                naive_tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        tokenized_prompts = torch.cat(
            [tokenized_prompts, naive_tokenized_prompts])
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS  # 用来将一个张量转换为一个模型的缓冲区，这样可以让这个张量在保存和加载模型时被保留，但在训练时不被更新
        self.register_buffer("token_suffix", embedding[:,
                                                       1 + n:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.csc = True
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.naive_embedding = naive_embedding.to(
            torch.device("cuda"))


    def forward(self):
        ctx = self.ctx
        ctx_dim = ctx.size(-1)
        dmx = self.domain_vectors  # dm 16 512
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_dm, -1, -1)  # dm 16 512
            if not self.csc:
                ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1,
                                              -1)  # dm cls 16 512
        else:
            ctx = ctx.unsqueeze(0).expand(self.n_dm, -1, -1,
                                          -1)  # dm cls 16 512

        dmx = dmx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)  # dm cls 16 512
        ctxdmx = torch.cat([ctx, dmx],
                           dim=2).reshape(self.n_cls * self.n_dm,
                                          self.n_ctx + self.n_dmx, ctx_dim)

        prefix = self.token_prefix
        suffix = self.token_suffix

        # naive
        neb = self.naive_embedding

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctxdmx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        prompts = torch.cat([prompts, neb], dim=0)

        return prompts



class Dacdl(nn.Module):
    def __init__(self, **config):
        super(Dacdl, self).__init__()

        drug_in_feats = 75
        drug_embedding = 128
        drug_padding = True
        drug_hidden_feats = [128, 128, 128]

        protein_emb_dim = 128
        num_filters = [128, 128, 128]
        kernel_size = [3, 6, 9]
        protein_padding = True
        ban_heads = 2

        mlp_in_dim = 256
        mlp_hidden_dim = 512
        mlp_out_dim = 512
        out_binary = 1

        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)

        self.protein_extractor = ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)
        self.bcn = weight_norm(
            BANLayer(v_dim=drug_hidden_feats[-1], q_dim=num_filters[-1], h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None)
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)


        clip_model = load_clip_to_cpu()  # 加载模型
        clip_model.float()
        self.prompt_learner = PromptLearner(clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype


    def forward(self, bg_d, v_p, mode="train"):
        v_d = self.drug_extractor(bg_d)  # 对药物进行GCN encoder
        v_p = self.protein_extractor(v_p)  # 对蛋白进行CNN encoder
        f, att = self.bcn(v_d, v_p)  # f, att 它们分别表示一个批次的药物分子和蛋白质之间的相互作用的得分和注意力权重
        score = self.mlp_classifier(f)
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        score = score / score.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1,keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * score @ text_features.t()
        return logits

class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding: # 表示是否对不同大小的药物分子进行补齐，使得它们的图结构具有相同的维度
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1] # 图卷积网络的输出特征的维度，即最后一层的维度

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1) # 第二个维度和第一个维度交换，得到一个新的张量
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=512):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, 512)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x