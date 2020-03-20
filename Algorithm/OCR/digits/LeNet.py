import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch
import numpy as np

class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        self.conv1_1 = nn.Sequential(     # input_size=(1*28*28)
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),   # padding=2保证输入输出尺寸相同
        )
        self.BN = nn.BatchNorm2d(1, momentum=0.5)
        self.conv1_2 = nn.Sequential(  # input_size=(16*28*28)
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),  # padding=2保证输入输出尺寸相同
            nn.ReLU(),  # input_size=(16*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2)  # output_size=(16*14*14)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3),
        )  # output 32*12*12
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3),
            nn.ReLU(),  # input_size=(32*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(8*5*5)
        )
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self._set_init(self.fc1)
        self.fc2 = nn.Linear(128, 11)
        self._set_init(self.fc2)
        self.softmax = nn.LogSoftmax(dim=1)

    def _set_init(self, layer):  # 参数初始化
        init.normal_(layer.weight, mean=0., std=.1)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        # x = x.view(-1, 28, 28)
        # x = self.BN(x)
        # x = x.view(-1, 1, 28, 28)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)

class rgbNet(nn.Module):
    def __init__(self, imtype):
        super(rgbNet, self).__init__()
        if imtype == 'bit':
            n = 1
        elif imtype == 'rgb':
            n = 3

        self.conv1_1 = nn.Sequential(     # input_size=(1*28*28)
            nn.Conv2d(
                in_channels=n,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),   # padding=2保证输入输出尺寸相同
        )
        self.BN = nn.BatchNorm2d(1, momentum=0.5)
        self.conv1_2 = nn.Sequential(  # input_size=(16*28*28)
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),  # padding=2保证输入输出尺寸相同
            nn.ReLU(),  # input_size=(16*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2)  # output_size=(16*14*14)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3),
        )  # output 32*12*12
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3),
            nn.ReLU(),  # input_size=(32*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(8*5*5)
        )
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self._set_init(self.fc1)
        self.fc2 = nn.Linear(128, 11)
        self._set_init(self.fc2)
        self.softmax = nn.LogSoftmax(dim=1)

    def _set_init(self, layer):  # 参数初始化
        init.normal_(layer.weight, mean=0., std=.1)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        # x = x.view(-1, 28, 28)
        # x = self.BN(x)
        # x = x.view(-1, 1, 28, 28)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)
class bitNet(nn.Module):
    def __init__(self,imtype='bit'):
        super(bitNet, self).__init__()
        if imtype == 'bit':
            n = 1
        elif imtype == 'rgb':
            n = 3  
        self.conv1 = nn.Conv2d(n, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 11)
    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
class newRgbNet(nn.Module):
    def __init__(self,imtype):
        super(newRgbNet, self).__init__()
        if imtype == 'bit':
            n = 1
        elif imtype == 'rgb':
            n = 3
        self.conv1_1 = nn.Sequential(     # input_size=(3*28*28)
            nn.Conv2d(
                in_channels=n,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32)
        )
        self.conv1_2 = nn.Sequential(  # input_size=(32*28*28)
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),  # padding=2保证输入输出尺寸相同
            nn.LeakyReLU(0.1),  # 
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)  # output_size=(64*14*14)
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(128),     # output_size=(128*14*14)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3),   # output_size=(128*12*12)
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)  # output_size=(128*6*6)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 6 * 6, 128),
            # nn.Dropout(0.5),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(128, 11)
    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class switchNet(nn.Module):
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
        self.softmax = nn.LogSoftmax(dim=1)


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
        return self.softmax(x)

