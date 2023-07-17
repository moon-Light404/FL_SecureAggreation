from attack.attack import Attack
import numpy as np
import random
import torch

class Scale_Attack(Attack):
    def __init__(self):
        super(Scale_Attack, self).__init__()

    # 权重放大
    def attack(self, users):
        if len(users) == 0:
            return
        for u in users:
            for param in u.local_net.parameters():
                new = random.randint(2, 4) * param
                param.data = new.clone()
