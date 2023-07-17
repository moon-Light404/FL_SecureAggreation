# -*- coding: utf-8 -*-
# @Project: federated-learning-master
# @Author: dingjun
# @File name: sal_test
# @Create time: 2023/3/15 23:22
import tenseal as ts
import numpy as np

import Nets
import attack

bits_scale = 26
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
)
# set the scale
context.global_scale = pow(2, bits_scale)

# galois keys are required to do ciphertext rotations
context.generate_galois_keys()


# plain_vector = [i for i in range(10000)]
# encrypted_vector = ts.bfv_vector(context, plain_vector)
#
#
#
# add_result = encrypted_vector + [i for i in range(10000)]
# print(add_result.decrypt())
#
# sub_result = encrypted_vector - [1,2,3,4,5]
# print(sub_result.decrypt())
#
# mul_result = encrypted_vector * [1,2,3,4,5]
# print(mul_result)


def enc_div(enc_x_list):
    add_result = enc_x_list[0]
    for enc_x in enc_x_list:
        add_result = enc_x + add_result
    add_result = add_result - enc_x_list[0]
    return add_result * (1/3)


a1 = ts.ckks_vector(context, [1.0,2.0,3.0])
a2 = ts.ckks_vector(context, [4.0,5.0,6.0])
a = []
a.append(a1)
a.append(a2)
add_result = enc_div(a)
print(add_result.decrypt())

# model = Nets.MLP()
# w = attack.flatten_params(model.parameters())
#
# w1 = ts.ckks_vector(context, w)


