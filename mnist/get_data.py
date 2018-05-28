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

batch_size = 1000


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # normalize,
    ])

trainset = torchvision.datasets.MNIST(root='mnist', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4,
                                           pin_memory=True)
testset = torchvision.datasets.MNIST(root='mnist', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def save_img():

    train_features = np.zeros(shape=(60000, 1, 28, 28), dtype=np.float32)
    train_labels = np.zeros((60000,), dtype=np.int8)
    for i in range(1):
        for batch_idx, (data, target) in enumerate(train_loader):

            train_features[batch_idx * batch_size:(batch_idx + 1) * batch_size] = data.numpy()
            train_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size] = target.numpy()


    np.save('x_train.npy', train_features)
    np.save('y_train.npy', train_labels)
    del train_features
    del train_labels

def save_test():
    test_features = np.zeros(shape=(10000, 1, 28, 28), dtype=np.float32)
    test_labels = np.zeros((10000,), dtype=np.int8)
    for batch_idx, (data, target) in enumerate(test_loader):

        test_features[batch_idx * batch_size:(batch_idx + 1) * batch_size] = data.numpy()
        test_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size] = target.numpy()
    np.save('x_test.npy', test_features)
    np.save('y_test.npy', test_labels)
    del test_features
    del test_labels

save_img()
save_test()