#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 10:59:40 2018

@author: xueyunzhe
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os 


from torch.autograd import Variable
import time
import numpy as np
import pickle
import torch.backends.cudnn as cudnn
import sys




DATA = 'stl10'
model_name = str(sys.argv[1])
aug = int(sys.argv[2])
use_cuda = True
resume = False
save_path = DATA + '_checkpoint%d'%aug
best_acc = 0

train_transform = transforms.Compose(
        [
                transforms.RandomCrop(96, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.44671062,  0.43980984,  0.40664645), (0.26034098,  0.25657727,  0.27126738)),
                ])

test_transform = transforms.Compose(
                [
                transforms.ToTensor(),
                transforms.Normalize((0.44671062,  0.43980984,  0.40664645), (0.26034098,  0.25657727,  0.27126738)),
                ])

if aug == 0:
    train_transform = test_transform

trainset = torchvision.datasets.STL10(root='stl10', split='train', download=False, transform=train_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
testset = torchvision.datasets.STL10(root='stl10', split='test', download=False, transform=test_transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False, num_workers=8, pin_memory=True)



    

if model_name == 'resnet':
    from model.resnet import ResNet18
    net = ResNet18(10)
elif model_name == 'lenet':
    from model.lenet import LeNet
    net = LeNet(10)
elif model_name == 'densenet':
    from model.densenet import DenseNet
    net = DenseNet(growthRate=12, depth=40, reduction=0.5,
                        bottleneck=True, nClasses=10)
elif model_name == 'vgg':
    from model.vgg import VGG
    net = VGG('VGG16', num_classes=10)

if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(save_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(save_path + '/%s_ckpt.t7'%model_name)
    net.load_state_dict(checkpoint['net']) 
    
if use_cuda:
    Device = int(sys.argv[3])
#    Device = 0
    net.cuda(Device)
    cudnn.benchmark = True
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
        net.parameters(), 
        lr = 0.1,
        momentum = 0.9,
        weight_decay=1e-4,
        nesterov=True,
        )

def train(epoch):
    print('\nEpoch: %d' %epoch)
    net.train()
    train_loss = 0
    correct = 0
    a = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(Device), target.cuda(Device)
        data, target = Variable(data), Variable(target)
        
        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.data[0] * target.size(0)
        predicted = outputs.data.max(1)[1]
        correct += predicted.eq(target.data).cpu().sum()
        
    print('Train loss: %0.5f,     Train_accuracy: %0.5f' %(train_loss / len(train_loader.dataset), correct / len(train_loader.dataset)))
    print('This epoch cost %0.2f seconds' %(time.time() - a))

        
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    a = time.time()
    
    for batch_idx, (data, target) in enumerate(test_loader):
        if use_cuda:
            data, target = data.cuda(Device), target.cuda(Device)
        data, target = Variable(data), Variable(target)
        outputs = net(data)
        loss = criterion(outputs, target)
        
        test_loss += loss.data[0] * target.size(0)
        predicted = outputs.data.max(1)[1]
        correct += predicted.eq(target.data).cpu().sum()
        
    print('Test loss: %0.5f,     Test_accuracy: %0.5f' %(test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)))
    print('This epoch cost %0.2f seconds' %(time.time() - a))
       
    acc = correct / len(test_loader.dataset)
    if acc > best_acc:
        print('Saving...')
        state = {
                'net': net.state_dict(),
                }

        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        torch.save(state, save_path + '/%s_ckpt.t7'%model_name)
        best_acc = acc
            

start_epoch = 1

if model_name == 'resnet' or model_name == 'vgg':
    for epoch in range(start_epoch, start_epoch + 300):
        if epoch <150:
            lr = 0.1
        elif epoch < 250:
            lr = 0.01
        else:
            lr = 0.001
        optimizer.param_groups[0]['lr'] = lr
        train(epoch)
        test(epoch)
        print('Learning rate: %f' % optimizer.param_groups[0]['lr'])
        
if model_name == 'densenet':
    for epoch in range(start_epoch, start_epoch + 300):
        if epoch <150:
            lr = 0.1
        elif epoch < 225:
            lr = 0.01
        else:
            lr = 0.001
        optimizer.param_groups[0]['lr'] = lr
        train(epoch)
        test(epoch)
        print('Learning rate: %f' % optimizer.param_groups[0]['lr'])
    
if model_name == 'lenet':
    for epoch in range(start_epoch, start_epoch + 200):
        if epoch <81:
            lr = 0.05
        elif epoch < 122:
            lr = 0.005
        else:
            lr = 0.0005
        optimizer.param_groups[0]['lr'] = lr
        train(epoch)
        test(epoch)
        print('Learning rate: %f' % optimizer.param_groups[0]['lr'])