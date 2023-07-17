from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json

from get_datasets import get_dataset, mnist_iid, DatasetSplit

dataset_train, dataset_test = get_dataset('./data', 'mnist')
dict_users = mnist_iid(dataset_train, 10)
idxs = dict_users[0]
train_ldr = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=32, shuffle=True)
test_ldr = DataLoader(dataset_test, batch_size=32, shuffle=True)
for images, labels in train_ldr:
    images, labels = images,labels
    print(images)
    print(labels)
    break