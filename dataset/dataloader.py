import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer


class DTIDataset(data.Dataset):
    def __init__(self, df, max_drug_nodes = 290):  # df:pandas.DataFrame对象，表示训练集的数据表格
        self.df = df
        self.max_drug_nodes = max_drug_nodes

        self.atom_featurizer = CanonicalAtomFeaturizer()  # 提取原子的特征，比如原子类型，电荷，杂化类型等
        self.bond_featurizer = CanonicalBondFeaturizer(
            self_loop=True)  # 是提取键的特征，比如键类型，是否是共轭键，是否是环键等,这个类的参数self_loop表示是否为分子图添加自环，也就是每个节点和自己相连的边，这里设置为True，表示添加自环
        self.fc = partial(smiles_to_bigraph,
                          add_self_loop=True)  # 它的作用是创建一个偏函数，并赋值给self.fc这个属性，这个偏函数的作用是将SMILES字符串转换成分子图，它使用了smiles_to_bigraph这个函数，并将其中的参数add_self_loop固定为True，表示为分子图添加自环

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        v_d = self.df.iloc[index]['SMILES']
        v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]  # 表示实际节点数
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d = v_d.add_self_loop()

        v_p = self.df.iloc[index][4:]

        y = self.df.iloc[index]["LABEL"]
        # y = torch.Tensor([y])
        return v_d, v_p, y  # 返回处理好的药物向量，蛋白质向量

def df_load(type):
    drug_data = pd.read_csv('F:\drugCancer_Code\DACDRPL\data_process\data_sets\GDSC\data\drugandsmile.csv')
    rna_data = pd.read_csv("F:\drugCancer_Code\DACDRPL\data_process\data_sets\GDSC\data\cnv_abs_copy_number_picnic_20191101.csv").T.iloc[2:,2:]
    rna_data['SANGER_MODEL_ID'] = rna_data.index
    if type == 'train':
        response = pd.read_csv('F:\drugCancer_Code\DACDRPL\data_process\data_sets\GDSC\data/response_LUAD.csv')
    elif type == "test":
        response = pd.read_csv('F:\drugCancer_Code\DACDRPL\data_process\data_sets\GDSC\data/response_SCLC.csv')

    response['LABEL'] = ''
    response.loc[response["LN_IC50"] <= 0.1, 'LABEL'] = 1
    # response.loc[response["LN_IC50"] <= response["MAX_CONC"], 'LABEL'] = 1
    response.loc[response["LN_IC50"] > 0.1, 'LABEL'] = 0
    # response.loc[response["LN_IC50"] > response["MAX_CONC"], 'LABEL'] = 0

    df = pd.DataFrame()
    df['DRUG_NAME'] = ''
    df['SANGER_MODEL_ID'] = ''
    df['LABEL'] = ''

    df['DRUG_NAME'] = response['DRUG_NAME']
    df['SANGER_MODEL_ID'] = response['SANGER_MODEL_ID']
    df['LABEL'] = response['LABEL']

    df1 = pd.merge(df, drug_data, how='inner', on='DRUG_NAME')
    df2 = pd.merge(df1, rna_data, how='inner', on='SANGER_MODEL_ID')

    # del df2['Unnamed: 0']
    del df2['index']


    print('value_counts',df2['LABEL'].value_counts())

    # df2.to_pickle("data_process/dat                                                                                                                                                                                                                                                                                                                                                                                                      a_sets/GDSC/data_test/test_data.pkl")

    # df = pd.read_pickle("data_process/data_sets/GDSC/data_test/test_data.pkl")

    return df2

def shuffled(df2):
    df_true = df2[df2['LABEL'] == 1]
    df_false = df2[df2['LABEL'] == 0]
    df_f = df_false.sample(1618, axis=0)
    df_1b1 = pd.concat([df_true, df_f])
    shuffled_df = df_1b1.sample(frac=1)
    shuffled_df.to_pickle("dataset/1b1_LUAD.pkl")