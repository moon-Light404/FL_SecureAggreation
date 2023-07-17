

import numpy as np
import random
import torch

from attack.attack import Attack


class Byzantine_Attack(Attack):
    def __init__(self):
        super(Byzantine_Attack, self).__init__()

    # 加入噪声
    def attack(self, users):
        if len(users) == 0:
            return
        for u in users:
            for key, val in u.local_net.state_dict().items():
                martrix = torch.Tensor(np.random.normal(0, 0.8, val.shape))
                val.add_(martrix.cuda())
