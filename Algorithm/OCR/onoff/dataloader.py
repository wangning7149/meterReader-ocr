import torch

import torch.utils.data.dataset as Dataset

import torch.utils.data.dataloader as DataLoader
from torch.autograd import Variable
import numpy as np
import os
Data = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]])

Label = np.asarray([[0], [1], [0], [2]])
# weight = torch.Tensor([1, 2, 1, 1, 10])
# loss_fn = torch.nn.CrossEntropyLoss( weight=weight)
# input = Variable(torch.randn(3, 5))  # (batch_size, C)
# target = Variable(torch.FloatTensor(3).random_(5))
# loss = loss_fn(input, target.long())
# print(input)
# print(target)
# print(loss)
import cv2

# 创建子类

class subDataset(Dataset.Dataset):

    # 初始化，定义数据内容和标签

    def __init__(self,):
        self.data = []
        # todo 加载数据集以及制作标签
        self.Data1_dir = os.listdir("./data/0")
        self.Data2_dir = os.listdir("./data/1")
        for it in self.Data1_dir:
            try:
                image = cv2.imread("./data/0/"+it)

            except:
                print(it)
            image = cv2.resize(image,(40,40))
            self.data.append([image,0])

        for it in self.Data2_dir:
            try:
                image = cv2.imread("./data/1/"+it)

            except:
                print(it)
            image = cv2.resize(image,(40,40))
            self.data.append([image,1])




    # 返回数据集大小

    def __len__(self):
        return len(self.data)

    # 得到数据内容和标签

    def __getitem__(self, index):
        d = self.data[index]

        return d[0],d[1]


# if __name__ == '__main__':
#     dataset = subDataset()
#
#     print('dataset大小为：', dataset.__len__())
#
#     # 创建DataLoader迭代器
#     dataloader = DataLoader.DataLoader(dataset, batch_size=3, shuffle=True, num_workers=1)
#     for i, item in enumerate(dataloader):
#         print('i:', i)
#         data, label = item
#         print('data:', data.size())
#         print('label:', label)
