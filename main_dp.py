# -*- coding: utf-8 -*-
# @Project: federated-learning-master
# @Author: dingjun
# @File name: main
# @Create time: 2023/3/12 19:45
import torch

from attack.Drift_attack import Drift_Attack
import attack
import get_datasets
from Server import Server
from User import User
from attack.flipping_attack import Flipp_Attack
from get_datasets import DatasetSplit
from Nets import *
import numpy as np
import copy
import matplotlib.pyplot as plt
import tenseal as ts

from main import DP
from utils import row_into_parameters
import warnings
import time
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    torch.manual_seed(73)

    conf = get_datasets.get_config('./conf.json')
    dataset_train, dataset_test = get_datasets.get_dataset("./data",conf["type"])
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

    w_glob = server.net.state_dict() # 获取服务器的模型参数
    w_locals = [w_glob for i in range(conf["no_clients"])]
    # # 是否聚合所有客户端的梯度
    # if conf["all_clients"]:
    #     print("Aggregation over all clients") # 聚合所有客户的梯度
    #     w_locals = [w_glob for i in range(conf["no_clients"])]
    sum_time = 0
    # 拉普拉斯噪声
    dp = DP(1, 0.2)
    # 开始联邦学习
    for epoch in range(conf["global_epochs"]):
        loss_locals = []
        # 如果不是聚合所有客户的梯度，每一个epoch后聚合时都要清空w_locals
        # if not conf["all_clients"]:
        #     w_locals = []
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
            w, loss = users[idx].local_train_dp(copy.deepcopy(server.net.state_dict()), dp)
            w_locals[idx] = copy.deepcopy(w)
            # print(w_locals[idx])
            loss_locals.append(copy.deepcopy(loss))
        if conf["attack"]:
            attacker.attack(mallicious_users) # 恶意用户进行梯度攻击
            for u in mallicious_users:
                w_locals[u.user_id] = copy.deepcopy(u.local_net.state_dict())

        time_start = time.time()
        # 服务器进行聚合
        w_glob = server.FedAvg(w_locals)
        time_end = time.time()
        sum_time += (time_end - time_start)
        server.net.load_state_dict(w_glob)

        # 服务器进行密文聚合
        # w_param = server.FedAvg_enc(w_locals).decrypt()
        # row_into_parameters(w_param, server.net.parameters())

        # 求每一轮的平均损失(大Epoch)
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(epoch, loss_avg))
        loss_train.append(loss_avg)


        acc, loss = server.model_test()
        acc_test.append(acc)
        loss_test.append(loss)
        print("Testing accuracy: {:.2f}".format(acc))
        print("Testing accuracy: {:.2f}".format(loss))

        # 对服务器进行test, 每隔3轮进行测试
        # if epoch % 3 == 0 or epoch == conf["global_epochs"]:
        #     acc, loss = server.model_test()
        #     acc_test.append(acc)
        #     loss_test.append(loss)
    # 写入数据到文件中
    # f = open("./TestResult/2/info.txt", 'a', encoding='GBK')
    # for key, val in conf.items():
    #     f.write(key + ":" + str(val) + "\n")
    # f.write("loss_train:  ")
    # for lo in loss_test:
    #     f.write(str(lo) + "  ")
    # f.write("\n acc_test:  ")
    # for acc in acc_test:
    #     f.write(str(float(acc)) + "  " )
    # f.write("总用时s：" + str(sum_time) + " \n")
    # f.write("平均用时s" + str(sum_time/conf["global_epochs"]))
    # f.close()
    # 画图
    # plt.plot(range(len(loss_train)), loss_train)
    # plt.xlabel("epoch")
    # plt.ylabel('train_loss')
    # # plt.savefig('TestResult/2/1.png')
    # plt.show()


    plt.plot(range(len(acc_test)), acc_test)
    plt.xlabel("epoch")
    plt.ylabel('acc_test')
    # plt.savefig('TestResult/2/2.png')
    plt.show()
    for i in range(len(acc_test)):
        print("{:.2f},".format(acc_test[i]))
