'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight, mode='fan_out')
#        if m.bias:
#            init.constant(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal(m.weight)
#        if m.bias:
#            init.constant(m.bias, 0)
        
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

        self.apply(_weights_init)
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
