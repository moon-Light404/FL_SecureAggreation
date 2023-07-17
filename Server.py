# -*- coding: utf-8 -*-
# @Project: federated-learning-master
# @Author: dingjun
# @File name: Server
# @Create time: 2023/3/12 19:44
from collections import defaultdict

from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import copy

import utils
from utils import get_cosine_similarity



class Server:
    def __init__(self, model, conf, dataset):
        self.net = model
        self.conf = conf
        self.test_dataset = dataset
        self.test_ldr = DataLoader(dataset, batch_size=64, shuffle=True)

    def get_shape(self):
        p_shape = {}
        for key, val in self.net.state_dict().items():
            p_shape[key] = val.shape
        return p_shape

    # G是w列表中的下标，处理对应下标的数据
    def FedAvg_list(self, w, pos_list):
        w_avg = copy.deepcopy(w[pos_list[0]])
        for key in w_avg.keys():
            for i in pos_list:
                w_avg[key] += w[i][key].cuda()
            w_avg[key] -= w[pos_list[0]][key].cuda()
            w_avg[key] = torch.div(w_avg[key], len(pos_list))
        return w_avg


    def FedAvg(self, w):
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key].cuda()
            w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg

    # 返回密态下模型的和
    def FedAvg_enc(self, w):
        w_glob = copy.deepcopy(w[0])
        for key in w_glob.keys():
            for i in range(1, len(w)):
                w_glob[key] = np.add(w_glob[key], w[i][key])  # 加密态
        return w_glob

    # 计算模型权重之间的距离：对于每对参与者i和j，计算其本地模型权重向量之间的欧氏距离。
    # 计算每个参与者的距离和：一共有n个参与者，对于每个参与者i，假设有f个攻击者，计算参与者与其他最近的n - f - 1
    # 个参与者模型权重之间的距离和。
    # 选择距离和最小的模型：在所有参与者中，找到距离和最小的模型作为聚合模型。

    def create_distance(self, w):
        distances = defaultdict(dict)
        num = 0
        for k in w[0].keys():
            if num == 0:
                for i in range(len(w)):
                    for j in range(i):
                        distances[i][j] = distances[j][i] = np.linalg.norm(
                            w[i][k].cpu().numpy() - w[j][k].cpu().numpy())
                num = 1
            else:
                for i in range(len(w)):
                    for j in range(i):
                        distances[j][i] += np.linalg.norm(w[i][k].cpu().numpy() - w[j][k].cpu().numpy())
                        distances[i][j] += distances[j][i]
        return distances

    def krum(self, w, user_count, mal_count, return_index=False, distances=None):
        global minimal_error_index
        non_malicious_count = user_count - mal_count - 1 # non_malicious_count = user_count - mal_count
        if distances is None:
            # 初始化距离
            distances = self.create_distance(w)
        minimal_error = 1e20
        for user in distances.keys():
            score = sorted(distances[user].values())
            cur_score = sum(score[:non_malicious_count])
            if cur_score < minimal_error:
                minimal_error = cur_score
                minimal_error_index = user
        if return_index: # 只返回下标
            return minimal_error_index
        else:
            return w[minimal_error_index]


    # 计算余弦相似度
    def create_cosine_similarity(self, x, y):
        similar = 0
        for k in x.keys():
            similar += get_cosine_similarity(x[k].cpu().numpy(), y[k].cpu().numpy())
        return similar

    def trimmed_mean(self, w, user_count, mal_count):
        num_to_consider = user_count - mal_count
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys(): # 遍历参数名,weight, bias
            tmp = []
            for i in range(len(w)):
                tmp.append(w[i][k].cpu().numpy()) # 提取每一列出来
            tmp = np.array(tmp)
            med = np.median(tmp, axis=0)
            new_tmp = []
            for i in range(len(tmp)):
                new_tmp.append(tmp[i] - med)
            new_tmp = np.array(new_tmp)
            # 记录近的索引
            good_vals = np.argsort(abs(new_tmp), axis=0)[:num_to_consider]
            # 根据从数组中索引取值
            good_vals = np.take_along_axis(new_tmp, good_vals, axis=0)
            k_weight = np.array(np.mean(good_vals) + med)
            w_avg[k] = torch.from_numpy(k_weight).cuda()
        return w_avg

    # Bulyan是目前SOTA的一个拜占庭容错算法，它十分巧妙，简单地来说，就是不断循环选择，然后跑一次Trimmed
    # Mean。而特别的是，该算法是使用Krum来选择的。所以，目前SOTA的容错算法是Krum + Trimmed
    # Mean的一个结合。

    def mkrum(self, w, user_count, mal_count):
        # assert user_count >= 4 * mal_count + 3
        set_size = user_count - 2 * mal_count - 2
        selection_set = []

        distances = self.create_distance(w)
        while len(selection_set) < set_size:
            selected_index = self.krum(w, user_count -
                                       len(selection_set),
                                       mal_count, True, distances)
            selection_set.append(w[selected_index])
            # 直接从distances数组除去这个下标所对应的用户距离
            distances.pop(selected_index)
            # 遍历其他客户，删除到select_index客户的距离
            for remain_user in distances.keys():
                distances[remain_user].pop(selected_index)
        return self.FedAvg(np.array(selection_set))
        # return self.trimmed_mean(np.array(selection_set), len(selection_set), 0)
        # return self.trimmed_mean(np.array(selection_set), len(selection_set), mal_count)

    # 服务器模型评估
    def model_test(self):
        self.net.eval()
        test_loss = 0
        correct = 0
        for idx, (data, target) in enumerate(self.test_ldr):
            data, target = data.cuda(), target.cuda()
            log_probs = self.net(data)
            # 测试loss的和
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # 得到索引最大的值
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            # 正确分类的个数
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        test_loss /= len(self.test_ldr.dataset)
        # 计算准确率
        accuracy = round(float(100.00 * correct / len(self.test_ldr.dataset)), 3)
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_ldr.dataset), accuracy))
        return accuracy, test_loss


    def AFA(self, w, user_count, mal_count):
        B = set() # Bad Clients
        G = set([i for i in range(user_count)]) # Good Clients
        R = {-1}
        Asize = user_count - 2 * mal_count
        alpha = 12
        while len(R) > 0:
            if len(G) <= Asize:
                break
            R.clear()
            w1 = self.FedAvg_list(w, list(G))
            cross_similarity = [0.0 for i in range(user_count)] # 先将相似值初始为0
            for k in G: # k 是客户端的编号
                cs = self.create_cosine_similarity(w1, w[k])
                cross_similarity[k] = cs
            mean = np.mean(cross_similarity) # 平均值
            median = np.median(cross_similarity) # 中位数
            std = np.std(cross_similarity, ddof=1) # 标准差
            if mean < median: # 筛选恶意客户端
                for k in G:
                    if cross_similarity[k] < median - alpha * std:
                        R = R.union({k})
                        G = G - {k}
            else:
                for k in G:
                    if cross_similarity[k] > median + alpha * std:
                        R = R.union({k})
                        G = G - {k}
            alpha = alpha + 0.3
            B = B.union(R)
        return self.FedAvg_list(w, list(G))
        # last_cross = 1000000
        # min = 0
        # if len(G) == 1:
        #     return w[G[0]]
        # else:
        #     w1 = self.FedAvg_list(w, G)
        #     for k in G:
        #         cs_2 = self.create_cosine_similarity(w1, w[k])
        #         if abs(cs_2) < last_cross:
        #             min = k
        #             last_cross = cs_2
        # return w[min]



    def defend(self, defence_method, w, user_count, mal_count):
        if defence_method == 'AFA':
            return self.AFA(w, user_count, mal_count)
        elif defence_method == 'Krum':
            return self.krum(w, user_count, mal_count)
        elif defence_method == 'Trimmed_mean':
            return self.trimmed_mean(w, user_count, mal_count)
        elif defence_method == 'Mkrum':
            return self.mkrum(w, user_count, mal_count)
        else:
            return self.FedAvg(w)







