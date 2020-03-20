import sys
import os
import cv2
import torch.utils.data.dataloader as DataLoader
import numpy as np
import torch
import torch.optim as optim

from Algorithm.OCR.onoff.dataloader import subDataset
from Algorithm.OCR.onoff.model1 import switchNet
def train():
    dataset = subDataset()

    # todo 创建DataLoader迭代器
    dataloader = DataLoader.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1,)
    valloader = DataLoader.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1,)
    net = switchNet()
    net.train()
    lr = 0.001  # todo 学习率
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), weight_decay=1e-5, lr=lr)
    epoch = 2   # todo 迭代次数   （所有的数据训练一遍 称为 迭代一次）
    for e in range(epoch):
        for i,data in enumerate(dataloader):
            input = data[0]
            label = data[1]
            input = torch.tensor(input,dtype=torch.float32).permute((0,3,1,2))

            label = torch.tensor(label)
            optimizer.zero_grad()
            out = net(input)
            _, predicted = torch.max(out.data, 1)
            correct = (predicted == label.long()).sum()
            loss = criterion(out, label.long())

            loss.backward()
            optimizer.step()
            for i, data in enumerate(valloader):
                input = data[0]
                label = data[1]
                input = torch.tensor(input, dtype=torch.float32).permute((0, 3, 1, 2))

                label = torch.tensor(label)

                out = net(input)
                _, predicted = torch.max(out.data, 1)
                correct = (predicted == label.long()).sum()
                print(correct.item())
                break
    torch.save(net.state_dict(),"./left_switch_net.pth")  #todo  模型保存



if __name__ == '__main__':
    train()
