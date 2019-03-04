
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from models import LeNet, SubstituteModel
import os
import torch.backends.cudnn as cudnn
import time

DATA = 'mnist'

if DATA == 'mnist':
    train_dir = '/home/xyz/dataset/mnist'
    test_dir = '/home/xyz/dataset/mnist'


resume = True
use_cuda = True
dtype = torch.float32


best_acc = 0

batch_size = 128

seed = 2018
print('Random seed: ', seed)
torch.manual_seed(seed)
save_path = 'model'




test_transform = transforms.Compose(
        [
            # transforms.Pad(2),
            transforms.ToTensor(),

                ])
print('start normalize')
trainset = datasets.MNIST(root=train_dir, train=True, download=True, transform=test_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4,
                                           pin_memory=True)
testset = datasets.MNIST(root=test_dir, train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


net = LeNet()
criterion = nn.CrossEntropyLoss()


if use_cuda:
    print('start move to cuda')
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

    device = torch.device("cuda:0")
    net.to(device=device)
    criterion.to(device=device, dtype=dtype)




optimizer = optim.Adam(
    net.parameters(),
    lr=0.001,
)


def train(epoch):
    # global monitor
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    a = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)
        optimizer.zero_grad()
        outputs= net(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = outputs.data.max(1)[1]
        correct += predicted.eq(target.data).sum().item()

    print('Train loss: %0.5f,     Train_accuracy: %0.5f' % (
        train_loss / len(train_loader.dataset), correct / len(train_loader.dataset)))
    print('This epoch cost %0.2f seconds' % (time.time() - a))

    return correct / len(train_loader.dataset)


def test(epoch):
    global best_acc
    # monitor
    net.eval()
    test_loss = 0
    correct = 0
    a = time.time()

    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(test_loader):
            if use_cuda:
                data, target = data.to(device=device, dtype=dtype), target.to(device=device)

            outputs= net(data)
            loss = criterion(outputs, target)

            test_loss += loss.item()
            predicted = outputs.data.max(1)[1]
            correct += predicted.eq(target.data).sum().item()

        print('Test loss: %0.5f,     Test_accuracy: %0.5f' % (
            test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)))
        print('This epoch cost %0.2f seconds' % (time.time() - a))

    acc = correct / len(test_loader.dataset)
    if acc > best_acc:
        print('Saving...')
        state = {
            'net': net.state_dict(),
        }

        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        torch.save(state, save_path + '/lenet.t7')
        best_acc = acc

    return acc




def main():
    start_epoch = 1
    for epoch in range(start_epoch, start_epoch + 20):

        train(epoch)
        test(epoch)

        print('Learning rate: %f' % optimizer.param_groups[0]['lr'])



if __name__ == '__main__':
    main()
