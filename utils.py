# torch转list加密
import functools
import math

import numpy as np
import torch
from phe import paillier

# 生成同态加密密钥
public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)
print("公钥：")
print(public_key)
print("私钥：")
print(private_key)



# 给梯度参数加噪声
def add_noise(parameters, dp):
    noise = None
    # 拉普拉斯1
    if dp.dp_mode == 1:
        noise = torch.tensor(np.random.laplace(0, dp.dp_val, parameters.shape)).cuda()
    # 高斯2
    else:
        noise = torch.cuda.FloatTensor(parameters.shape).normal_(0, dp.dp_val)
    return parameters.add_(noise)

# 加密
def encrypt_vector(public_key, parameters):
    parameters = parameters.flatten(0).cpu().numpy().tolist()
    parameters = [public_key.encrypt(parameter) for parameter in parameters]
    return parameters

# list解密
def decrypt_vector(private_key, parameters):
    parameters = [private_key.decrypt(parameter) for parameter in parameters]
    return parameters


# 两个模型之间的距离
# def model_dist(model_1, model_2):
#     dist_sum = 0
#     for name, layer in model_1.named_parameters():
#         dist_sum += torch.sum(torch.pow(layer.data - model_2.state_dict()[name].data, 2))
#     return math.sqrt(dist_sum)


def flatten_params(params):
    # return [i.data.cpu().numpy().flatten() for i in params]
    return np.concatenate([i.data.cpu().numpy().flatten() for i in params])


# 将row中的参数写进parameters中，==复制
def row_into_parameters(row, parameters):
    offset = 0
    for param in parameters:
        # new_size: param数组中元素的个数，多维变成一维
        new_size = functools.reduce(lambda x, y: x * y, param.shape)
        current_data = row[offset:offset + new_size]

        param.data[:] = torch.from_numpy(np.array(current_data).reshape(param.shape))
        offset += new_size


def zero_into_parameters(parameters):
    # 清空模型的参数
    for p in parameters:
        p.data[:] = torch.zeros_like(p).clone()


# 服务器加密全局模型的参数字典
def get_glob_net_enc(glob_parameters):
    for key, val in glob_parameters.items():
        glob_parameters[key] = encrypt_vector(public_key, val.clone())


# 客户端解密全局模型
def get_glob_dec(glob_parameters, p_shape, num):
    glob_temp = {}
    for var in glob_parameters:
        glob_temp[var] = decrypt_vector(private_key, glob_parameters[var])

        glob_temp[var] = torch.Tensor(glob_temp[var])
        glob_temp[var] = glob_temp[var].reshape(p_shape[var])
        # print(glob_temp[var])
        glob_temp[var] = (glob_temp[var].cuda() / num)
    return glob_temp


def get_cosine_similarity(x, y):
    t = np.sum(x * y)
    non = np.linalg.norm(x) * np.linalg.norm(y)
    return np.round(t / float(non), 5)


