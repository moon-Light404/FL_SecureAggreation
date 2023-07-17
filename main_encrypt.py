import time

import torch


import attack
import get_datasets
import utils
from Server import Server
from User import User
from attack.flipping_attack import Flipp_Attack
from get_datasets import DatasetSplit
from Nets import *
import numpy as np
import copy
import matplotlib.pyplot as plt
import warnings
from phe import paillier
warnings.filterwarnings('ignore')



if __name__ == '__main__':
    torch.manual_seed(73)

    conf = get_datasets.get_config('./conf.json')
    dataset_train, dataset_test = get_datasets.get_dataset('./data' ,conf["type"])
    # 每个用户的数据下标
    dict_users = get_datasets.mnist_iid(dataset_train, conf["no_clients"])
    # 生成服务器
    server = None
    if conf["model_name"] == "mlp":
        server = Server(MLP().cuda(), conf, dataset_test)
    elif conf["model_name"] == "cnn":
        server = Server(CNNMnist().cuda(), conf, dataset_test)

    # 恶意用户和正常用户的数量
    mal_count = int(conf["mal_frac"] * conf["no_clients"])
    avg_count = int(conf["frac"] * conf["no_clients"])
    # 恶意用户的id
    mal_user_id = np.random.choice(range(conf["no_clients"]), mal_count, replace = False)

    # 初始化攻击方式
    if conf["attack"]:
        # attacker = arbitra_attack.ArbitraAttack(0)
        # attacker = attack.DriftAttack(2)
        attacker = Flipp_Attack(0)

    users = []
    for usr_id in range(conf["no_clients"]):
        if usr_id in mal_user_id:
            is_mal = True
        else:
            is_mal = False
        users.append(
            User(model=copy.deepcopy(server.net).cuda(), conf=conf, dataset=dataset_train, idxs=dict_users[usr_id],
                 id=usr_id, is_mallicious=is_mal))
    # print(users[0].get_enc_net())

    # 恶意用户列表
    mallicious_users = [u for u in users if u.is_mallicious]

    loss_train = []
    loss_test, acc_test = [], []

    loading = False
    w_glob = server.net.state_dict() # 获取服务器的模型参数

    # 是否聚合所有客户端的梯度
    # if conf["all_clients"]:
    #     print("Aggregation over all clients") # 聚合所有客户的梯度
    #     w_locals = [w_glob for i in range(conf["no_clients"])]
    print(users[0].get_net_enc())
    # kk = utils.get_glob_dec(w_tmp, users[0].get_shape())
    # for k, v in kk.items():
    #     print(k)
    #     print("\n" * 5)
    #     print(v)


    # w_locals中的模型都是加密后的
    w_locals = [users[0].get_net_enc() for i in range(conf["no_clients"])]
    sum_time = 0 # 计算训练时间

    # 开始联邦学习
    for epoch in range(conf["global_epochs"]):
        time_start = time.time()
        loss_locals = []

        # 按比例计算选择的客户端
        m = max(avg_count, 1)
        select_users = np.random.choice(range(conf["no_clients"]), m, replace = False)
        for idx in select_users:
            print("   ==============User %d开始训练===============" %idx)
            if users[idx].is_mallicious:
                print("用户{}是恶意用户端".format(idx))
            #     w, loss = users[idx].local_train_mal(net=copy.deepcopy(server.net))
            else:
                print("用户{}是正常用户端".format(idx))
            # 客户端将加密后的模型转给服务器， loading=false第一次，true不是第一次
            w, loss = users[idx].local_train_enc(w_glob, loading)
            w_locals[idx] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))
        # if conf["attack"]:
        #     attacker.attack(mallicious_users) # 恶意用户进行梯度攻击
        #     for u in mallicious_users:
        #         w_locals[u.user_id] = copy.deepcopy(u.local_net.state_dict())

        # 服务器进行聚合,此时w_glob是加密态
        w_glob = server.FedAvg_enc(w_locals)


        # 第一轮的w_glob没有被加密
        loading = True
        # server.net.load_state_dict(w_glob)
        time_end = time.time()
        sum_time += time_end - time_start
        # 服务器进行密文聚合
        # w_param = server.FedAvg_enc(w_locals).decrypt()
        # row_into_parameters(w_param, server.net.parameters())

        # 求每一轮的平均损失(大Epoch)
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(epoch, loss_avg))
        loss_train.append(loss_avg)

        server.net.load_state_dict(copy.deepcopy(utils.get_glob_dec(w_glob, server.get_shape())), strict=True)
        acc, loss = server.model_test()
        print("Testing accuracy: {:.2f}".format(acc))
        print("Testing loss: {:.2f}".format(loss))

        # 对服务器进行test, 每隔3轮进行测试
        # if epoch % 3 == 0 or epoch == conf["global_epochs"]:
        #     acc, loss = server.model_test()
        #     acc_test.append(acc)
        #     loss_test.append(loss)
    print("同态加密FedAvg训练用时：{:.2f}".format(sum_time))
    # 画图
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 15}
    plt.title('同态加密下FedAvg聚合的准确率变化图', font)
    plt.plot(acc_test, marker='*')
    plt.xlabel("Epoch")
    plt.ylabel('LOSS OF TRAIN.%')
    plt.grid()
    plt.show()

    acc, loss = server.model_test()
    print("Testing accuracy: {:.2f}".format(acc))
    print("Testing loss: {:.2f}".format(loss))