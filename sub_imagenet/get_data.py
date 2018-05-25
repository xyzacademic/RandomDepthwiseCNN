import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import torchvision.models as models

from torch.autograd import Variable
import time
import numpy as np
import pickle
import torch.backends.cudnn as cudnn
import sys


batch_size = 10




transform = transforms.Compose(
                [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # normalize,
                ])

train_dataset = torchvision.datasets.ImageFolder(root='train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True,
                                           num_workers=8, pin_memory=True, drop_last=True)#12739

test_dataset = torchvision.datasets.ImageFolder(root='val', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False,
                                           num_workers=8, pin_memory=True)

def save_img():


    train_features = np.zeros(shape=(12730, 3, 224, 224), dtype=np.float32)
    train_labels = np.zeros((12730,), dtype=np.int8)
    for batch_idx, (data, target) in enumerate(train_loader):

        train_features[batch_idx * batch_size:(batch_idx + 1) * batch_size] = data.numpy()
        train_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size] = target.numpy()
    np.save('x_train.npy', train_features)
    np.save('y_train.npy', train_labels)
    del train_features
    del train_labels

def save_test():
    test_features = np.zeros(shape=(500, 3, 224, 224), dtype=np.float32)
    test_labels = np.zeros((500,), dtype=np.int8)
    for batch_idx, (data, target) in enumerate(test_loader):

        test_features[batch_idx * batch_size:(batch_idx + 1) * batch_size] = data.numpy()
        test_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size] = target.numpy()
    np.save('x_test.npy', test_features)
    np.save('y_test.npy', test_labels)
    del test_features
    del test_labels

save_img()
save_test()
