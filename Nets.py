# -*- coding: utf-8 -*-
# @Project: federated-learning-master
# @Author: dingjun
# @File name: Nets
# @Create time: 2023/3/12 19:26
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(784, 300)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout() # dropout防止模型过拟合提高效果，随机按一定改概率抛弃神经网络单元
        self.layer2 = nn.Linear(300, 10)

    def forward(self, x):
        x = x.reshape(-1, 28 * 28)
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# 张量的channles、输出张量的channles、卷积核大小、步长=1

class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        # self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])  # x.shape[0]自动计算，表示图片的数量
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 128, 3)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        return x



#
#
# import math
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib as mpl
#
# mpl.rcParams['axes.unicode_minus'] = False
#
# fig = plt.figure(figsize=(6, 4))
# ax = fig.add_subplot(111)
#
# x = np.arange(-10, 10)
# y = np.where(x < 0, 0, x)
#
# plt.xlim(-11, 11)
# plt.ylim(-11, 11)
#
# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')
#
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data', 0))
# ax.set_xticks([-10, -5, 0, 5, 10])
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data', 0))
# ax.set_yticks([-10, -5, 5, 10])
#
# plt.plot(x, y, label="ReLU", color="blue")
# plt.legend()
# plt.show()
