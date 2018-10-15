import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neural_network import MLPClassifier
import torch.utils.data
import torch.backends.cudnn as cudnn
import time
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, num_classes=8):
        super(MLP, self).__init__()
        self.bn0 = nn.BatchNorm1d(100000)
        self.fc1 = nn.Linear(100000, 1000)
        self.bn1 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        out = self.bn0(x)
        out = self.fc1(out)
        out = self.bn1(out)
        features = torch.sigmoid(out)
        out = self.fc2(features)

        return features





np.random.seed(2017)
os.chdir('temp_data')
# rh_origin
batch_size = 2
train_data = np.load('train_data_3_2_100000.npy').astype(np.float32)
train_label = np.load('train_label_3_2_100000.npy').astype(np.int64)
test_data = np.load('test_data_3_2_100000.npy').astype(np.float32)
test_label = np.load('test_label_3_2_100000.npy').astype(np.int64)

os.chdir(os.pardir)


train_data, train_label, test_data, test_label = torch.from_numpy(train_data), torch.from_numpy(train_label),torch.from_numpy(test_data), torch.from_numpy(test_label)
trainset = torch.utils.data.TensorDataset(train_data, train_label)
testset = torch.utils.data.TensorDataset(test_data, test_label)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)


seed = 2018
print('Random seed: ', seed)
torch.manual_seed(seed)
save_path = 'mlp_checkpoint'
dtype = torch.float32
resume = True
use_cuda = True
net = MLP(8)

criterion = nn.CrossEntropyLoss()

if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(save_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(save_path + '/ckpt.t7')
    net.load_state_dict(checkpoint['net'])

if use_cuda:
    print('start move to cuda')
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

    net = torch.nn.DataParallel(net, device_ids=[0,1])
    device = torch.device("cuda:0")
    net.to(device=device)
    criterion.to(device=device, dtype=dtype)



optimizer = optim.SGD(
    net.parameters(),
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0001,
    nesterov=True
)


def train(epoch):
    # global monitor
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    a = time.time()
    #    pred = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)

        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = outputs.max(1)[1]
        correct += predicted.eq(target).sum().item()

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
            predicted = outputs.max(1)[1]
            correct += predicted.eq(target).sum().item()

        print('Test loss: %0.5f,     Test_accuracy: %0.5f' % (
            test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)))
        print('This epoch cost %0.2f seconds' % (time.time() - a))

    acc = correct / len(test_loader.dataset)
    # if acc > best_acc:
    #     print('Saving...')
    #     state = {
    #         'net': net.module.state_dict(),
    #     }
    #
    #     if not os.path.isdir(save_path):
    #         os.mkdir(save_path)
    #     torch.save(state, save_path + '/ckpt.t7')
    #     best_acc = acc

    return acc

def save_features():
    net.eval()
    test_loss = 0
    correct = 0
    a = time.time()


    os.chdir('temp_data')

    train_data = np.zeros(shape=(430, 1000), dtype=np.float16)
    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(train_loader):
            if use_cuda:
                data, target = data.to(device=device, dtype=dtype), target.to(device=device)

            features = net(data)
            train_data[batch_idx*batch_size:(batch_idx+1)*batch_size] = features.cpu()
    np.save('train_rh.npy', train_data)

    train_data = np.zeros(shape=(8, 1000), dtype=np.float16)
    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(test_loader):
            if use_cuda:
                data, target = data.to(device=device, dtype=dtype), target.to(device=device)

            features = net(data)
            train_data[batch_idx*batch_size:(batch_idx+1)*batch_size] = features.cpu()
    np.save('test_rh.npy', train_data)


def main():
    start_epoch = 1
    for epoch in range(start_epoch, start_epoch + 9):

        row = {'epoch': str(epoch), 'train_acc': str(train(epoch)), 'test_acc': str(test(epoch))}
        # csv_logger.writerow(row)

        print('Learning rate: %f' % optimizer.param_groups[0]['lr'])
        if epoch in [60]:
            optimizer.param_groups[0]['lr'] *= 0.1

    print('Saving...')
    state = {
        'net': net.module.state_dict(),
    }

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    torch.save(state, save_path + '/ckpt.t7')



if __name__ == '__main__':
    save_features()