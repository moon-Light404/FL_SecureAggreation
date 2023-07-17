# -*- coding: utf-8 -*-
# @Project: federated-learning-master
# @Author: dingjun
# @File name: main_nn
# @Create time: 2023/3/16 0:50
import matplotlib

import get_datasets

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
from Nets import MLP, CNNMnist

def test(net_g, data_loader):
    # testing
    net_g.eval()
    test_loss = 0
    correct = 0
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

    return correct, test_loss


conf = get_datasets.get_config('./conf.json')
dataset_train, dataset_test = get_datasets.get_dataset('./data' ,conf["type"])
net_glob = CNNMnist()
net_glob.cuda()
optimizer = optim.SGD(net_glob.parameters(), lr=conf["lr"], momentum=conf["momentum"])
train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)

list_loss = []
net_glob.train()
for epoch in range(10):
    batch_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = net_glob(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        # 第50、100、150个batch时的loss
        batch_loss.append(loss.item())
    loss_avg = sum(batch_loss)/len(batch_loss)
    print('\nTrain loss:', loss_avg)
    list_loss.append(loss_avg)

plt.plot(range(len(list_loss)), list_loss)
plt.xlabel('epochs')
plt.ylabel('train loss')
plt.savefig('_{}_{}_{}.png'.format(conf["type"], conf["model_name"], 10))

test_acc, test_loss = test(net_glob, test_loader)
print("准确度:  %d" %test_acc)
print("损失值:  %d" %test_loss)