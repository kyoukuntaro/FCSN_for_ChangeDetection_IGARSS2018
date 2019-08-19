import numpy as np
import os

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from model import FC_EF

class OSCD(Dataset):
    def __init__(self, dir_nm):
        super(OSCD, self).__init__()
        self.dir_nm = dir_nm
        self.file_ls = os.listdir(dir_nm)
        self.file_size = len(self.file_ls)

    def __getitem__(self, idx):
        mat = np.load(self.dir_nm + self.file_ls[idx]).astype(np.float)
        x1 = mat[:3,:,:]/255
        x2 = mat[3:6,:,:]/255
        lbl = mat[6,:,:]/255
        return x1, x2, lbl

    def __len__(self):
        return self.file_size

def main():
    train_dir = './data/train/'
    test_dir = './data/test/'
    lr = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = OSCD(train_dir)
    train_dataloader = DataLoader(train_data, batch_size=10, shuffle=True)
    test_data = OSCD(test_dir)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
    model = FC_EF().to(device, dtype=torch.float)

    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        loss_v = []
        model.train()
        for i, data in enumerate(train_dataloader):
            x1, x2, lbl = data
            x1 = x1.to(device, dtype=torch.float)
            x2 = x2.to(device, dtype=torch.float)
            lbl = lbl.to(device, dtype=torch.long)
            y = model(x1,x2)
            optimizer.zero_grad()
            loss = criterion(y, lbl)
            loss.backward()
            optimizer.step()
            loss_v.append(loss.item())
            if(i%20==0 and i>0):
                print(np.mean(loss_v))
                loss_v = []

        loss_v = []
        model.eval()
        for i, data in enumerate(test_dataloader):
            x1, x2, lbl = data
            x1 = x1.to(device, dtype=torch.float)
            x2 = x2.to(device, dtype=torch.float)
            lbl = lbl.to(device, dtype=torch.long)
            y = model(x1, x2)
            optimizer.zero_grad()
            loss = criterion(y, lbl)
            loss.backward()
            optimizer.step()
            loss_v.append(loss.item())
        print('test:', np.mean(loss_v))
        loss_v = []


if __name__ == '__main__':
    main()