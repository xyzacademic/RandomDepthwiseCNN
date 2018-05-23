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

batch_size = 100


transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ])

trainset = torchvision.datasets.STL10(root='stl10', split='train', download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10,
                                           pin_memory=True)

testset = torchvision.datasets.STL10(root='stl10', split='test', download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True)


def save_img():


    train_features = np.zeros(shape=(5000, 3, 96, 96), dtype=np.float32)
    train_labels = np.zeros((5000,), dtype=np.int8)

    for batch_idx, (data, target) in enumerate(train_loader):
        train_features[batch_idx * batch_size:(batch_idx + 1) * batch_size] = data.numpy()
        train_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size] = target.numpy()


    np.save('x_train.npy', train_features)
    np.save('y_train.npy', train_labels)
    del train_features
    del train_labels

def save_test():
    test_features = np.zeros(shape=(8000, 3, 96, 96), dtype=np.float32)
    test_labels = np.zeros((8000,), dtype=np.int8)
    for batch_idx, (data, target) in enumerate(test_loader):

        test_features[batch_idx * batch_size:(batch_idx + 1) * batch_size] = data.numpy()
        test_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size] = target.numpy()
    np.save('x_test.npy', test_features)
    np.save('y_test.npy', test_labels)
    del test_features
    del test_labels

save_img()
save_test()
