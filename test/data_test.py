# -*- coding: utf-8 -*-
# @Project: federated-learning-master
# @Author: dingjun
# @File name: data_test
# @Create time: 2023/3/16 20:59
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import get_datasets
from get_datasets import DatasetSplit
import matplotlib.pyplot as plt



trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
dataset_train = datasets.CIFAR10('./data/cifar/', train=True, download=True, transform=trans_mnist)
dataset_test = datasets.CIFAR10('./data/cifar/', train=False, download=True, transform=trans_mnist)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# print(len(dataset_train))
# img, label = dataset_train[0]
# print(img)
# print(label)

def get_count_lit(dataset):
    class_count = [0 for _ in range(10)]
    for item in dataset:
        img, label = item
        class_count[label] += 1
    return class_count


def get_graph(user_id, count_list):
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.bar(x = classes, height=classes, color = 'steelblue')
    plt.ylabel('数量')
    plt.title('用户{}的数据分布图'.format(user_id))
    for x, y in enumerate(count_list):
        plt.text(x, y+0.1, y, ha='center')
    plt.show()

k = get_count_lit(dataset_test)
print(k)


# torch.manual_seed(73)
# # 每个用户的数据下标
# dict_users = get_datasets.mnist_iid(dataset_train, 20)
#
# for i in range(20):
#     count = get_count_lit(dict_users[i], dataset_train)
#     get_graph(i, count)
# dataloader = DataLoader(dataset_train, batch_size=32, shuffle=False)
# imgs, labels = next(iter(dataloader))
# img = imgs[0]
# img[:, :5, :5] = 2.8
# print(imgs[0])

# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.tight_layout()
#     imgs[i][:, :5, :5] = 2.8
#     plt.imshow(imgs[i][0], cmap='gray', interpolation='none')#子显示
#     plt.title("Ground Truth: {}".format(labels[i])) #显示title
#     plt.xticks([])
#     plt.yticks([])
# plt.show()