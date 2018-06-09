#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 18:58:51 2018

@author: xueyunzhe
"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import sys

DATA = 'stl10'
source_model = sys.argv[1]
target_model = sys.argv[2]


aug = 1
use_cuda = True
resume = True
save_path = DATA + '_checkpoint%d'%aug
batch_size = 50

if DATA == 'cifar10':
    test_transform = transforms.Compose(
                    [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                    ])

    testset = torchvision.datasets.CIFAR10(root='cifar10', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

if DATA == 'stl10':
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.44671062, 0.43980984, 0.40664645), (0.26034098, 0.25657727, 0.27126738)),
        ])

    testset = torchvision.datasets.STL10(root='stl10', split='test', download=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False, num_workers=8, pin_memory=True)


def get_model(model_name):
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
    
    return net


source_net = get_model(source_model)
target_net = get_model(target_model)

if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(save_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(save_path + '/%s_ckpt.t7'%source_model)
    source_net.load_state_dict(checkpoint['net']) 
    checkpoint = torch.load(save_path + '/%s_ckpt.t7'%target_model)
    target_net.load_state_dict(checkpoint['net']) 

if use_cuda:
#    Device = int(sys.argv[3])
    Device = 0
    source_net.cuda(Device)
    target_net.cuda(Device)
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()

def test(eps):
    
    source_net.train()
    target_net.eval()
    correct = 0
    
    for batch_idx, (data, target) in enumerate(test_loader):
        if use_cuda:
            data, target = data.cuda(Device), target.cuda(Device)
        data, target = Variable(data, requires_grad=True), Variable(target)
        outputs = source_net(data)
        loss = criterion(outputs, target)
        loss.backward()
        
        grad_sign = data.grad.cpu().data.sign().numpy()
        data_ = data.cpu().data.numpy() + eps * grad_sign  
        # data_ = np.clip(data_, np.array([-1.98947368, -1.98436214, -1.71072797], dtype=np.float32).reshape((1, 3, 1, 1)),
        #                 np.array([ 2.05910931,  2.1308642 ,  2.12068966], dtype=np.float32).reshape((1, 3, 1, 1))) #cifar10
        data_ = np.clip(data_, np.array([-1.71586748, -1.71414186, -1.49906137], dtype=np.float32).reshape((1, 3, 1, 1)),
                        np.array([ 2.12524889,  2.18331951,  2.18733837], dtype=np.float32).reshape((1, 3, 1, 1)))#stl10
        data_ = Variable(torch.from_numpy(data_).cuda(Device))
        outputs = target_net(data_)
        
        predicted = outputs.data.max(1)[1]
        correct += predicted.eq(target.data).cpu().sum()
        
    print('Test_accuracy: %0.5f' % (correct / len(test_loader.dataset)))
    
       
def save_adv(eps):
    source_net.train()
    
    test_features = np.zeros(shape=(8000, 3, 96, 96), dtype=np.float32)
    for batch_idx, (data, target) in enumerate(test_loader):
        if use_cuda:
            data, target = data.cuda(Device), target.cuda(Device)
        data, target = Variable(data, requires_grad=True), Variable(target)
        outputs = source_net(data)
        loss = criterion(outputs, target)
        loss.backward()
        
        grad_sign = data.grad.cpu().data.sign().numpy()
        data_ = data.cpu().data.numpy() + eps * grad_sign  
        # data_ = np.clip(data_, np.array([-1.98947368, -1.98436214, -1.71072797]).reshape((1, 3, 1, 1)),
        #                 np.array([ 2.05910931,  2.1308642 ,  2.12068966]).reshape((1, 3, 1, 1))) #cifar10
        data_ = np.clip(data_, np.array([-1.71586748, -1.71414186, -1.49906137]).reshape((1, 3, 1, 1)),
                        np.array([ 2.12524889,  2.18331951,  2.18733837]).reshape((1, 3, 1, 1)))#stl10
        data_ = data_ * np.array([0.26034098, 0.25657727, 0.27126738]).reshape((1, 3, 1, 1)) + np.array([0.44671062, 0.43980984, 0.40664645]).reshape((1, 3, 1, 1))
        test_features[batch_idx*batch_size:(batch_idx+1)*batch_size] = data_
    
    np.save('adv/test_data_%s%d.npy'%(source_model, aug), test_features)
        
        
print('Source: %s     Target: %s'%(source_model, target_model))
for eps in [0.0625]:
    print('eps %s:'%eps)
    # test(eps)
    save_adv(eps)
    
print('\n\n')
