# -*- coding: utf-8 -*-
# @Project: federated-learning-master
# @Author: dingjun
# @File name: attack
# @Create time: 2023/3/12 21:42
import copy
import functools
import math

import numpy as np
import torch

import get_datasets
from Nets import MLP
from User import User
from utils import row_into_parameters, flatten_params, zero_into_parameters

# 清零
class Attack(object):
    def __init__(self):
        super(Attack, self).__init__()

    def attack(self, users):
        if len(users) == 0:
            return
        # 清空用户的参数
        for u in users:
            zero_into_parameters(u.local_net.parameters())


# conf = get_datasets.get_config()
# dataset_train, dataset_test = get_datasets.get_dataset(conf["type"])
# # 每个用户的数据下标
# dict_users = get_datasets.mnist_iid(dataset_train, conf["no_clients"])
# users = []
# id = 0
# for i in range(3):
#     u = User(model=copy.deepcopy(MLP()), conf=get_datasets.get_config(), dataset=dataset_train, idxs=dict_users[i],
#                              id=i)
#     users.append(u)
#
# zero_into_parameters(users[1].local_net.parameters())
#
# for i in users[1].local_net.parameters():
#     print(i)


# attack = Attack(0.0001)
# attack.attack(users)
# print(next(iter(users[0].local_net.parameters())))
# print(next(iter(users[1].local_net.parameters())))
