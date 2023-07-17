# -*- coding: utf-8 -*-
# @Project: federated-learning-master
# @Author: dingjun
# @File name: arbitra_attack
# @Create time: 2023/3/13 10:27
import copy

import numpy as np
import torch

import Nets
from attack.attack import Attack

from utils import zero_into_parameters, row_into_parameters, flatten_params


class User:
    def __init__(self, model):
        self.local_net = model


class Drift_Attack(Attack):
    def __init__(self, data_std):
        super(Drift_Attack, self).__init__()
        self.data_std = data_std

    def attack_grads(self, m, s):
        m[:] += self.data_std * s
        return m


    def attack(self, users):
        if len(users) == 0 or self.data_std == 0:
            return
        user_grads = []
        for usr in users:
            usr_grad = flatten_params(usr.get_net())
            user_grads.append(usr_grad)
        grad_mean = np.mean(user_grads, axis=0)
        grad_stdv = np.var(user_grads, axis=0) ** 0.5

        mal_grads = self.attack_grads(grad_mean, grad_stdv)

        for usr in users:
            row_into_parameters(mal_grads, usr.local_net.parameters())

    # def attack(self, users):
    #     model = copy.deepcopy(users[0].local_net)
    #     mal_model = {}
    #     for key, val in model.state_dict().items():
    #         user_key_grads = []
    #         for usr in users:
    #             usr_tmp = usr.local_net.state_dict()[key]
    #             usr_tmp = flatten_params(usr_tmp).reshape(val.shape)
    #             user_key_grads.append(usr_tmp)
    #         m = np.mean(user_key_grads, axis=0)
    #         s = np.std(user_key_grads, axis=0)
    #
    #         mal_model[key] = torch.from_numpy(self.attack_grads(m, s))
    #     for k, v in mal_model.items():
    #         print(v)
    #     for u in users:
    #         u.local_net.load_state_dict(mal_model)


# net = Nets.CNNMnist()
#
# torch.set_printoptions(precision=15)
# attack = Drift_Attack(2.0)
# user0 = User(net)
# user1 = User(net)
# user2 = User(net)
# users = []
# users.append(user0)
# users.append(user1)
# users.append(user2)
#
# for k, v in user0.local_net.state_dict().items():
#     print(k)
#     print(v)
# print("\n" * 20)
# attack.attack(users)



# for k, v in user0.local_net.state_dict().items():
#     print(k)
#     print(v)





