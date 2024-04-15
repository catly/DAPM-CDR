from dataset.dataloader import *
import dgl
from torch.utils.data import DataLoader
from trainer import *
from model import *

def graph_collate_func(x):
    d, p, y = zip(*x)
    d = dgl.batch(d)
    return d, torch.tensor(np.array(p).astype(float)), torch.tensor(y)

def main():

    seed = 6
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = 'cuda'
    # train_data_df = df_load('train')
    train_data_df = pd.read_pickle("dataset/1b1_LUAD.pkl")[:3200]
    train_dataset = DTIDataset(train_data_df)

    # test_data_df = df_load('test')
    test_data_df = pd.read_pickle("dataset/1b1_SCLC.pkl")[:3200]
    test_dataset = DTIDataset(test_data_df)

    params = {'batch_size': 16, 'shuffle': True, 'num_workers': 0,
              'drop_last': True, 'collate_fn': graph_collate_func}

    training_generator = DataLoader(train_dataset, **params)
    testing_generator = DataLoader(test_dataset, **params)

    model = Dacdl().to(device)

    for name, param in model.named_parameters():
        if "text_encoder" in name:
            param.requires_grad_(False)

    opt_x = torch.optim.Adam(model.parameters(), lr=0.3)
    opt_u = torch.optim.Adam(model.parameters(), lr=0.0001)
    trainer = Trainer(model, device,opt_x, opt_u, training_generator,testing_generator)
    result = trainer.train()
    return result




if __name__ == '__main__':
    main()


