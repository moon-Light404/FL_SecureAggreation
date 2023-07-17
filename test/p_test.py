import numpy as np
import torch

import Nets
import random
# model = Nets.MLP()
#
# for key, val in model.state_dict().items():
#     print(key)
#     print(val)

# a = [1,2,3,4,5]
# print(a[:3])


# a = [[1,2,],
#      [3,4]]
# a1 = [[1,2],
#       [3,2]]
#
# b = []
# b.append(a)
# b.append(a1)
# print(np.median(b, axis=0))


# def cosSim(x, y):
#     tmp = np.sum(x*y)
#     non = np.linalg.norm(x) * np.linalg.norm(y)
#     return np.round(tmp/float(non), 5)
#
#
# #
# if __name__=='__main__':
#     a=np.array([1,2])
#     b=np.array([2,4])
#     sim=cosSim(a,b)
#     print(sim)
#
# R = [0]
# print(R[0])

# a = [1, 2, 3]
# print(np.median(a))

# a = [1, 2, 3]
# a[:] = 6
# print(a)

# ------------------------
# net = Nets.CNNMnist()
#
# for param in net.parameters():
#     print(param)
#
# for param in net.parameters():
#     new = random.randint(2, 5) * param
#     param.data = new.clone()
#
# print("\n" * 5)
#
# for param in net.parameters():
#     print(param)
# -----------------------------
# net = Nets.CNNMnist()

# for param in net.parameters():
#     print(param)
# for key, val in net.state_dict().items():
#     print(key)
#     # ans = torch.Tensor(np.random.normal(0, 20, val.shape))
#     # val = 0
# print("\n" * 5)

# for param in net.parameters():
#     print(param)


a = 1
b = 2
c = 1
c += b - a
print(c)
