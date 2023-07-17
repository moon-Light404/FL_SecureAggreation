# -*- coding: utf-8 -*-
# @Project: federated-learning-master
# @Author: dingjun
# @File name: flipping_attack
# @Create time: 2023/3/17 20:56
import copy

import torch

from Nets import CNNMnist
from attack.attack import Attack


class Flipp_Attack(Attack):
    def __init__(self):
        super(Flipp_Attack, self).__init__()

    # 梯度取反
    def attack(self, users):
        if len(users) == 0:
            return
        for u in users:
            for param in u.local_net.parameters():
                new_param = -1 * param
                param.data = new_param.clone()


