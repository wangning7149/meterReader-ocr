import torch.nn as nn
from torch.nn import init
import torch
import torch.nn.functional as F
import numpy as np
class switchNet(nn.Module):  # todo 搭建模型
    def __init__(self):
        super(switchNet, self).__init__()
        self.conv1_1 = nn.Sequential(     # input_size=(3*40*40)
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
        )
        self.BN = nn.BatchNorm2d(16, momentum=0.5)
        self.conv1_2 = nn.Sequential(  # input_size=(16*28*28)
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
            ),
            nn.ReLU(),  # input_size=(16*38*38)
            nn.MaxPool2d(kernel_size=2, stride=2)  # output_size=(16*19*19)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3),
            nn.ReLU(),  # input_size=(32*17*17)
            nn.MaxPool2d(2, 2)  # output_size=(32*8*8)
        )

        self.fc1 = nn.Linear(32 * 8* 8, 64)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self._set_init(self.fc1)
        self.fc2 = nn.Linear(64, 2)
        self._set_init(self.fc2)


    def _set_init(self, layer):  # 参数初始化
        init.normal_(layer.weight, mean=0., std=.1)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        # x = x.view(-1, 28, 28)
        # x = self.BN(x)
        # x = x.view(-1, 1, 28, 28)
        x = self.conv1_1(x)
        x = self.BN(x)
        x = self.conv1_2(x)
        x = self.conv2_1(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
if __name__ == '__main__':
    data = torch.from_numpy(np.ones([1,3,40,40])).float()

    wn = switchNet()
    re = wn(data)
    print(re.size())

