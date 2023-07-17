# -*- coding: utf-8 -*-
# @Project: federated-learning-master
# @Author: dingjun
# @File name: get_datasets
# @Create time: 2023/3/12 19:27

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import json


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label  # 返回图像和标签


def mnist_iid(dataset, num_users):
    torch.manual_seed(73)
    num_items = int(len(dataset) / num_users)  # 每个用户分得的数据量
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        # 从全部的索引中取出 dist_users[i]的数据索引,set取差集
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users  # 所有用户的索引

def cifar_iid(dataset, num_users):
    torch.manual_seed(73)
    num_items = int(len(dataset)/num_users) # 每次选择num_items个
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]  # [0,1,2,3...200]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)  # 0-60000的列表list
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  # 堆叠成二维数据
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs中从左到右的labels是 0-9对应的数据索引
    idxs = idxs_labels[0, :]


    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users  # 返回用户的数据索引数组


# 获得测试集和数据集
def get_dataset(directory, name):
    print(name)
    if name == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(directory + '/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(directory + '/mnist/', train=False, download=True, transform=trans_mnist)
        return dataset_train, dataset_test
    elif name == 'cifar10':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(directory + '/cifar/', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10(directory + '/cifar/', train=False, download=True, transform=trans_cifar)
        return dataset_train, dataset_test


def get_config(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        file_content = f.read()
        load_dict = json.loads(file_content)
        return load_dict

# conf = get_config('./conf.json')
# # print(conf["lr"])
# f = open("./TestResult/1/info.txt", 'a')
# for key, val in conf.items():
#     f.write(key + ":" + str(val) + "\n")