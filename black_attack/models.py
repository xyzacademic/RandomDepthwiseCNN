import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
           init.constant_(m.bias, 0)

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.apply(_weights_init)

    def forward(self, x):
        # x = F.pad(x, 2)
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class SubstituteModel(nn.Module):

    def __init__(self, num_classes=10):
        super(SubstituteModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, num_classes)

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

class CNNModel(nn.Module):

    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, num_classes)

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

class StlModel(nn.Module):

    def __init__(self, num_classes=10):
        super(StlModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.fc3 = nn.Linear(256, num_classes)
        self.mean = torch.from_numpy(np.array([0.44671062, 0.43980984, 0.40664645]).reshape((1,3,1,1))).float().cuda()
        self.std = torch.from_numpy(np.array([0.26034098, 0.25657727, 0.27126738]).reshape((1,3,1,1))).float().cuda()

        self.apply(_weights_init)

    def forward(self, x):
        x = (x - self.mean) / self.std
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv4(out))
        out = F.avg_pool2d(out, 12)
        out = out.view(out.size(0), -1)
        out = self.fc3(out)
        return out

class LinearModel(nn.Module):
    def __init__(self, num_classes=10):
        super(LinearModel, self).__init__()
        self.linear1 = nn.Linear(3*96*96, 200)
        self.linear2 = nn.Linear(200,200)
        self.linear3 = nn.Linear(200, num_classes)
        self.apply(_weights_init)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        return out




def _weights_init_(m):
    #    classname = m.__class__.__name__
    #    print(classname)
    if isinstance(m, nn.Conv2d):
        state = 'normal'
        if state == 'normal':
            init.normal_(m.weight, 0, 1)
            init.normal_(m.bias, 0, 1)
        elif state == 'uniform':
            init.uniform_(m.weight, -1, 1)
            init.uniform_(m.bias, -1, 1)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=0.01)
        init.normal_(m.bias, std=0.01)
        # init.xavier_normal_(m.weight)


class Block(nn.Module):

    def __init__(self, n, k, bias=True):
        super(Block, self).__init__()
        self.pool = nn.AvgPool2d(2,2,0)
        self.conv = nn.Conv2d(n, n, k, groups=n,
                              padding=(k-1)//2,
                              bias=bias
                              )
        self.conv2 = nn.Conv2d(n, n, k, groups=n,
                              padding=(k-1)//2,
                               bias=bias
                              )
        # self.conv3 = nn.Conv2d(n, n, k, groups=n,
        #                        padding=(k - 1) // 2,
        #                       )
        self.apply(_weights_init_)

    def forward(self, x):
        out = self.pool(x)
        out = self.conv(out).sign_()
        out = self.conv2(out).sign_()

        return out


class Rh(nn.Module):

    def __init__(self, num_layers=3, kernel_size=3, n=1000, block=Block):
        super(Rh, self).__init__()
        self.kernel_size = kernel_size
        self.n = n
        self.num_layers = num_layers

        self.conv = nn.Conv2d(3, self.n, self.kernel_size,
                              padding=1,
                              )
        self.conv2 = self._make_layer(block, num_layers)

        self.apply(_weights_init_)

    def _make_layer(self, block, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(block(self.n, self.kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x).sign_()
        out = self.conv2(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)

        return out

if __name__ == '__main__':
    x = torch.randn((2,3, 96, 96))
    net = StlModel(10)
    output = net(x)
    print(output.size())
