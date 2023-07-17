# -*- coding: utf-8 -*-
# @Project: federated-learning-master
# @Author: dingjun
# @File name: User
# @Create time: 2023/3/12 19:24
import copy
import random

import torch
from torch import nn
from torch.utils.data import DataLoader

import utils
from get_datasets import DatasetSplit


class User:
    def __init__(self, model, conf, dataset=None, idxs=None, id=1,
                 is_mallicious=False):
        self.local_net = model
        self.conf = conf
        self.user_id = id
        self.train_dataset = dataset
        # 是否是恶意用户
        self.is_mallicious = is_mallicious
        self.loss_func = nn.CrossEntropyLoss()
        self.train_ldr = DataLoader(DatasetSplit(dataset, idxs),
                                    batch_size=self.conf["batch_size"],
                                    shuffle=True)

        # DataLoader加载数据集

    def set_net(self, net_dict):
        self.local_net.load_state_dic(net_dict)

    def get_net(self):
        return self.local_net.parameters()  # 返回每一层的参数，生成器迭代

    # 加密后的梯度参数字典
    def get_net_enc(self):
        local_param = {}
        for key, var in self.local_net.state_dict().items():
            local_param[key] = utils.encrypt_vector(utils.public_key, var)
        return local_param

    def get_net_dp(self, dp):
        local_param = {}
        for key, val in self.local_net.state_dict().items():
            local_param[key] = val.clone()
            local_param[key] = utils.add_noise(local_param[key], dp)
        return local_param

    def get_shape(self):
        p_shape = {}
        for key, val in self.local_net.state_dict().items():
            p_shape[key] = val.shape
        return p_shape

    # 接收服务器传来的梯度字典net_dict,正常训练
    def local_train(self, net_dict, poly_mode, dp, loading):
        # 同态加密
        if poly_mode == 2:
            # 第一轮服务器传过来的模型是明文形式的
            if loading:
                self.local_net.load_state_dict(copy.deepcopy(net_dict),
                                               strict=True)
            # 后面都是聚合之后的，和
            else:
                self.local_net.load_state_dict(copy.deepcopy(utils.get_glob_dec(net_dict, self.get_shape())),
                                               strict=True)
        # 明文形式和DP的模型
        else:
            self.local_net.load_state_dict(copy.deepcopy(net_dict))

        self.local_net.cuda()
        self.local_net.train()
        optimizer = torch.optim.SGD(self.local_net.parameters(), lr=self.conf["lr"], momentum=self.conf["momentum"])

        epoch_loss = []
        for iter in range(self.conf["local_epochs"]):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_ldr):
                images, labels = images.cuda(), labels.cuda()
                optimizer.zero_grad()
                log_probs = self.local_net(images)
                loss = self.loss_func(log_probs, labels)
                # 损失loss 向输入侧进行反向传播
                loss.backward()
                # optimizer.step()是优化器对
                #  的值进行更新，以随机梯度下降SGD 学习率(learning rate, lr)来控制步幅
                optimizer.step()
                # 输出本地训练的信息
                if batch_idx % 20 == 0:
                    # 200/1000 20%， loss
                    print('    Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.train_dataset),
                              100. * batch_idx / len(self.train_ldr), loss.item()))
                # 每个batch的loss, 64个loss的和
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        if poly_mode == 1:
            return self.local_net.state_dict(), sum(epoch_loss) / len(epoch_loss)
        elif poly_mode == 2:
            return self.get_net_enc(), sum(epoch_loss) / len(epoch_loss)
        else:
            return self.get_net_dp(dp), sum(epoch_loss) / len(epoch_loss)

    # 标签翻转攻击下的训练
    def local_train_label_mal(self, net_dict, poly_mode, dp, loading):
        # 同态加密
        if poly_mode == 2:
            # 第一轮服务器传过来的模型是明文形式的
            if loading:
                self.local_net.load_state_dict(copy.deepcopy(net_dict),
                                               strict=True)
            # 后面都是第二次聚合之后的，和
            else:
                self.local_net.load_state_dict(copy.deepcopy(utils.get_glob_dec(net_dict, self.get_shape())),
                                               strict=True)
        # 明文形式的模型
        else:
            self.local_net.load_state_dict(copy.deepcopy(net_dict))

        self.local_net.load_state_dict(copy.deepcopy(net_dict))
        self.local_net.cuda()
        self.local_net.train()
        optimizer = torch.optim.SGD(self.local_net.parameters(), lr=self.conf["lr"], momentum=self.conf["momentum"])

        epoch_loss = []
        # k = self.conf["poison_label"]
        for iter in range(self.conf["local_epochs"]):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_ldr):
                images, labels = images.cuda(), labels.cuda()
                # images[:][:, :10, :10] = 2.8
                labels[:] = 0
                optimizer.zero_grad()
                log_probs = self.local_net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                optimizer.step()

                # 输出本地训练的信息
                if batch_idx % 20 == 0:
                    # 200/1000 20%， loss
                    print('    Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.train_dataset),
                              100. * batch_idx / len(self.train_ldr), loss.item()))
                # 每个batch的loss, 64个loss的和
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        if poly_mode == 1:
            return self.local_net.state_dict(), sum(epoch_loss) / len(epoch_loss)
        elif poly_mode == 2:
            return self.get_net_enc(), sum(epoch_loss) / len(epoch_loss)
        else:
            return self.get_net_dp(dp), sum(epoch_loss) / len(epoch_loss)


    def local_train_enc(self, glob_net_sum, loading):
        if loading:
            self.local_net.load_state_dict(copy.deepcopy(utils.get_glob_dec(glob_net_sum, self.get_shape())),
                                           strict=True)
        else:
            self.local_net.load_state_dict(copy.deepcopy(glob_net_sum), strict=True)

        self.local_net.cuda()
        self.local_net.train()
        optimizer = torch.optim.SGD(self.local_net.parameters(), lr=self.conf["lr"], momentum=self.conf["momentum"])

        epoch_loss = []
        for iter in range(self.conf["local_epochs"]):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_ldr):
                images, labels = images.cuda(), labels.cuda()
                optimizer.zero_grad()
                log_probs = self.local_net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                optimizer.step()

                # 输出本地训练的信息
                if batch_idx % 20 == 0:
                    # 200/1000 20%， loss
                    print('    Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                        iter, batch_idx * len(images), len(self.train_dataset),
                              100. * batch_idx / len(self.train_ldr), loss.item()))
                # 每个batch的loss, 64个loss的和
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # 返回加密的模型
        return self.get_net_enc(), sum(epoch_loss) / len(epoch_loss)

        # 接收服务器传来的梯度字典net_dict

    def local_train_dp(self, net_dict, dp):
        self.local_net.load_state_dict(copy.deepcopy(net_dict))
        self.local_net.cuda()
        self.local_net.train()
        optimizer = torch.optim.SGD(self.local_net.parameters(), lr=self.conf["lr"], momentum=self.conf["momentum"])

        epoch_loss = []
        for iter in range(self.conf["local_epochs"]):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_ldr):
                images, labels = images.cuda(), labels.cuda()
                optimizer.zero_grad()
                log_probs = self.local_net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                optimizer.step()

                # 输出本地训练的信息
                if batch_idx % 20 == 0:
                    # 200/1000 20%， loss
                    print('    Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.train_dataset),
                              100. * batch_idx / len(self.train_ldr), loss.item()))
                # 每个batch的loss, 64个loss的和
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # 经过差分隐私处理后的模型梯度参数
        return self.get_net_dp(dp), sum(epoch_loss) / len(epoch_loss)

    # def local_train_malicious(self, net):
    #     self.local_net.load_state_dict(copy.deepcopy(net.state_dict()))
    #     self.local_net.cuda()
    #     self.local_net.train()
    #     optimizer = torch.optim.SGD(self.local_net.parameters(), lr=self.conf["lr"], momentum=self.conf["momentum"])
    #     epoch_loss = []
    #     for iter in range(self.conf["local_epochs"]):
    #         batch_loss = []
    #         for batch_idx, (images, labels) in enumerate(self.train_ldr):
    #             images, labels = images.cuda(), labels.cuda()
    #
    #             for k in range(self.conf["poison_per_batch"]):
    #                 labels[k] = self.conf['poison_label']
    #
    #             optimizer.zero_grad()
    #             output = self.local_net(images)
    #
    #             class_loss = self.loss_func(output, labels)
    #             dist_loss = model_dist(self.local_net, net)
    #             # print("dist_loss的值为:{}".format(dist_loss))
    #             loss = self.conf["alpha"] * class_loss + (1 - self.conf["alpha"] * dist_loss)
    #             loss.backward()
    #
    #             optimizer.step()
    #             if batch_idx % 20 == 0:
    #                 # 200/1000 20%， loss
    #                 print('    Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #                     iter, batch_idx * len(images), len(self.train_dataset),
    #                           100. * batch_idx / len(self.train_ldr), loss.item()))
    #             # 每个batch的loss, 64个loss的和
    #             batch_loss.append(loss.item())
    #         epoch_loss.append(sum(batch_loss) / len(batch_loss))
    #     mal_dict = dict()
    #     for name, data in self.local_net.state_dict().items():
    #         mal_dict[name] = self.conf["eta"] * (data - net.state_dict()[name]) + net.state_dict()[name]
    #     return mal_dict, sum(epoch_loss) / len(epoch_loss)

    # 接收服务器传来的梯度字典net_dict
    # def local_train(self, net_dict):
    #     self.local_net.load_state_dict(copy.deepcopy(net_dict))
    #     self.local_net.cuda()
    #     self.local_net.train()
    #     optimizer = torch.optim.SGD(self.local_net.parameters(), lr=self.conf["lr"], momentum=self.conf["momentum"])
    #
    #     epoch_loss = []
    #     for iter in range(self.conf["local_epochs"]):
    #         batch_loss = []
    #         for batch_idx, (images, labels) in enumerate(self.train_ldr):
    #             images, labels = images.cuda(), labels.cuda()
    #             optimizer.zero_grad()
    #             log_probs = self.local_net(images)
    #             loss = self.loss_func(log_probs, labels)
    #             loss.backward()
    #
    #             optimizer.step()
    #
    #             # 输出本地训练的信息
    #             if batch_idx % 20 == 0:
    #                 # 200/1000 20%， loss
    #                 print('    Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #                     iter, batch_idx * len(images), len(self.train_dataset),
    #                           100. * batch_idx / len(self.train_ldr), loss.item()))
    #             # 每个batch的loss, 64个loss的和
    #             batch_loss.append(loss.item())
    #         epoch_loss.append(sum(batch_loss) / len(batch_loss))
    #     return self.local_net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # def local_train_mal(self, net_dict):
    #     self.local_net.load_state_dict(copy.deepcopy(net_dict))
    #     self.local_net.cuda()
    #     self.local_net.train()
    #     optimizer = torch.optim.SGD(self.local_net.parameters(), lr=self.conf["lr"], momentum=self.conf["momentum"])
    #
    #     epoch_loss = []
    #     k = self.conf["poison_label"]
    #     for iter in range(self.conf["local_epoch"]):
    #         batch_loss = []
    #         for batch_idx, (images, labels) in enumerate(self.train_ldr):
    #             images, labels = images.cuda(), labels.cuda()
    #             # images[:][:, :10, :10] = 2.8
    #             labels[:] = 0
    #             optimizer.zero_grad()
    #             log_probs = self.local_net(images)
    #             loss = self.loss_func(log_probs, labels)
    #             loss.backward()
    #             optimizer.step()
    #             # 输出本地训练的信息
    #             if batch_idx % 20 == 0:
    #                 # 200/1000 20%， loss
    #                 print('    Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #                     iter, batch_idx * len(images), len(self.train_dataset),
    #                           100. * batch_idx / len(self.train_ldr), loss.item()))
    #             # 每个batch的loss, 64个loss的和
    #             batch_loss.append(loss.item())
    #         epoch_loss.append(sum(batch_loss) / len(batch_loss))
    #
    #     return self.local_net.state_dict(), sum(epoch_loss) / len(epoch_loss)
