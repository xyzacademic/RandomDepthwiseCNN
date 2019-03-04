import torch
import numpy as np
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models import LeNet, CNNModel, LinearModel, Rh
from torch.utils.data.sampler import SubsetRandomSampler
import torch.backends.cudnn as cudnn
import time
import torch.optim as optim
import os
import sys

class Oracle(object):
    def __init__(self, model, save_path='None', svm_path='None', device=None):
        self.device = device
        self.model = model
        self.save_path = save_path
        if os.path.exists(save_path):
            self.model.load_state_dict(torch.load(self.save_path)['net'])
            print('Load weights successfully for %s' % self.save_path)
        else:
            print('Initialized weights')
        self.model.to(device=device)
        from sklearn.svm import LinearSVC
        import pickle
        with open(svm_path, 'rb') as f:
            self.svc = pickle.load(f)

    def get_loader(self, x=None, y=None, batch_size=10, shuffle=False):
        assert isinstance(x, torch.Tensor)
        if y is None:
            y = torch.full(size=(x.size(0),), fill_value=-1).long()
        dataset = torch.utils.data.TensorDataset(x, y)
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                   num_workers=2, pin_memory=True)

    def predict(self, x, batch_size):
        self.get_loader(x, batch_size=batch_size, shuffle=False)
        self.model.eval()
        pred = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data = data.to(device=self.device, dtype=dtype)

                outputs = self.model(data).cpu().numpy()

                pred.append(torch.from_numpy(self.svc.predict(outputs)))

        return torch.cat(pred).long()

    def eval(self, x, y, batch_size):
        self.get_loader(x, y, batch_size=batch_size, shuffle=False)
        self.model.eval()

        correct = 0
        a = time.time()

        with torch.no_grad():

            for batch_idx, (data, target) in enumerate(self.data_loader):

                data, target = data.to(device=self.device, dtype=dtype), target.to(device=self.device)

                outputs = self.model(data).cpu().numpy()
                predicted = torch.from_numpy(self.svc.predict(outputs)).to(device=self.device)
                # outputs = self.model(data)
                # predicted = outputs.data.max(1)[1]
                correct += predicted.eq(target.data).sum().item()

            print('Test_accuracy: %0.5f' % (correct / len(self.data_loader.dataset)))
            print('This epoch cost %0.2f seconds' % (time.time() - a))


class Substitute(object):

    def __init__(self, model, save_path='None', device=None):
        self.device = device
        self.model = model
        self.save_path = save_path
        if os.path.exists(save_path):
            self.model.load_state_dict(torch.load(self.save_path)['net'])
            print('Load weights successfully for %s' % self.save_path)
        else:
            print('Initialized weights')
        self.model.to(device=device)

    def get_loader(self, x=None, y=None, batch_size=100, shuffle=False):
        assert isinstance(x, torch.Tensor)
        if y is None:
            y = torch.full(size=(x.size(0),), fill_value=-1).long()
        dataset = torch.utils.data.TensorDataset(x, y)
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                   num_workers=2, pin_memory=True)

    def predict(self, x, batch_size):
        self.get_loader(x, batch_size=batch_size, shuffle=False)
        self.model.eval()
        pred = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data = data.to(device=self.device, dtype=dtype)

                outputs = self.model(data)
                pred.append(outputs.data.max(1)[1])

        return torch.cat(pred).cpu()

    def eval(self, x, y, batch_size):
        self.get_loader(x, y, batch_size=batch_size, shuffle=False)
        self.model.eval()

        correct = 0
        a = time.time()

        with torch.no_grad():

            for batch_idx, (data, target) in enumerate(self.data_loader):

                data, target = data.to(device=self.device, dtype=dtype), target.to(device=self.device)

                predicted = self.model(data).max(1)[1]
                # outputs = self.model(data)
                # predicted = outputs.data.max(1)[1]
                correct += predicted.eq(target.data).sum().item()

            print('Test_accuracy: %0.5f' % (correct / len(self.data_loader.dataset)))
            # print('This epoch cost %0.2f seconds' % (time.time() - a))

    def train(self, x, y, batch_size, n_epoch):
        self.get_loader(x, y, batch_size, True)
        self.model.train()

        optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=0.001,
                )
        criterion = nn.CrossEntropyLoss().to(device=self.device)
        for epoch in range(n_epoch):
            train_loss = 0
            correct = 0
            a = time.time()

            for batch_idx, (data, target) in enumerate(self.data_loader):

                data, target = data.to(device=self.device, dtype=dtype), target.to(device=self.device)
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predicted = outputs.data.max(1)[1]
                correct += predicted.eq(target.data).sum().item()

            # print('Epoch #%d'%epoch)
            # print('Train loss: %0.5f,     Train_accuracy: %0.5f' % (
            #     train_loss / len(self.data_loader.dataset), correct / len(self.data_loader.dataset)))
            # print('This epoch cost %0.2f seconds' % (time.time() - a))

    def get_grad(self, x, y):
        self.get_loader(x, y, batch_size=1, shuffle=False)
        grads = []
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data = data.to(device=self.device, dtype=dtype).requires_grad_()
            outputs = self.model(data)[0, target]
            outputs.backward()
            grads.append(data.grad.cpu())
        return torch.cat(grads, dim=0)

    def get_loss_grad(self, x, y):
        self.get_loader(x, y, batch_size=100, shuffle=False)
        grads = []

        self.model.train()
        criterion = nn.CrossEntropyLoss().to(device=self.device)
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(device=self.device, dtype=dtype).requires_grad_(),\
                           target.to(device=self.device)
            outputs = self.model(data)
            loss = criterion(outputs, target)
            loss.backward()
            grads.append(data.grad.cpu())
        return torch.cat(grads, dim=0)

def get_data():
    train_dir = test_dir = '/home/y/yx277/research/ImageDataset/mnist'


    test_dataset = datasets.MNIST(root=test_dir, train=False,
        download=True, transform=transforms.ToTensor())

    indices = np.random.permutation(10000)
    sub_x = test_dataset.test_data[indices[:200]].float().reshape((-1, 1, 28, 28))
    sub_x /= 255
    test_data = test_dataset.test_data[indices[200:]].float().reshape((-1, 1, 28, 28))
    test_data /= 255
    test_label = test_dataset.test_labels[indices[200:]]

    return sub_x, test_data, test_label

def jacobian_augmentation(model, x_sub, y_sub, Lambda, samples_max):
    Lambda = np.random.choice([-1, 1])* Lambda
    x_sub_grads = model.get_grad(x=x_sub, y=y_sub)
    x_sub_new = x_sub + Lambda * torch.sign(x_sub_grads)
    if x_sub.size(0) <= samples_max / 2:
        return torch.cat([x_sub, x_sub_new], dim=0)
    else:
        return x_sub_new

def get_adv(model, x, y, epsilon):
    print('getting grads on epsilon=%.4f'%epsilon)
    grads = model.get_loss_grad(x, y)
    print('generating adversarial examples')
    return (x + epsilon * torch.sign(grads)).clamp_(0, 1)


def MNIST_bbox_sub(param, oracle_model, substitute_model, x_sub, test_data, \
                   test_label, aug_epoch, samples_max, n_epoch, fixed_lambda):

    for rho in range(aug_epoch):
        print('Epoch #%d:'%rho)
        # get x_sub's labels
        print('Current x_sub\'s size is %d'%(x_sub.size(0)))
        a = time.time()
        y_sub = oracle_model.predict(x=x_sub, batch_size=oracle_size)
        print('Get label for x_sub cost %.1f'%(time.time() - a))
        #train substitute model
        substitute_model.train(x=x_sub, y=y_sub, batch_size=128, n_epoch=n_epoch)
        #eval substitute on test data
        # substitute_model.eval(x=test_data, y=test_label, batch_size=128)

        if rho < param['data_aug'] - 1:
            print('Substitute data augmentation processing')
            a = time.time()
            x_sub = jacobian_augmentation(model=substitute_model, x_sub=x_sub, y_sub=y_sub, \
                                          Lambda=fixed_lambda, samples_max=samples_max)
            print('Augmentation cost %.1f seconds'%(time.time() - a))


        #Generate adv examples
        test_adv = get_adv(model=substitute_model, x=test_data, y=test_label, epsilon=param['epsilon'])
        # print('Substitute model FGSM attack itself\'s accuracy on adversarial samples #%d:' % (test_adv.size(0)))
        # substitute_model.eval(x=test_adv, y=test_label, batch_size=512)
        print('Oracle model FGSM attack\'s accuracy on adversarial samples #%d:' % (test_adv.size(0)))
        oracle_model.eval(x=test_adv, y=test_label, batch_size=oracle_size)
        torch.save(substitute_model.model.state_dict(), 'model/sub.t7')

if __name__ == '__main__':
    param = {
        'hold_out_size': 150,
        'test_batch_size': 128,
        'nb_epochs': 10,
        'learning_rate': 0.001,
        'data_aug': 20,
        'oracle_name': 'model/lenet',
        'epsilon': 0.0625,
        'lambda': 0.0625,

    }

    global seed, dtype, oracle_size
    oracle_size = 20
    dtype = torch.float32
    device = torch.device("cuda:0")
    device2 = torch.device("cuda:1")
    seed = 2018
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    sub_x, test_data, test_label = get_data()

    oracle_model = Substitute(model=LeNet(), save_path='model/lenet.t7', device=device)
#    oracle_model = Oracle(model=Rh(num_layers=1, kernel_size=7, n=100000),save_path='model/checkpoints_7_1/ckpt_100000.t7',\
#                                   svm_path='model/checkpoints_svm_7_1/svm.pkl', device=device)

    if sys.argv[1] =='0' or 'cnn':
        sub = CNNModel()
    else:
        sub = LinearModel()
    print(sys.argv[1])

    substitute_model = Substitute(model=sub, device=device2)
    MNIST_bbox_sub(param=param, oracle_model=oracle_model, substitute_model=substitute_model, \
                   x_sub=sub_x, test_data=test_data, test_label=test_label, aug_epoch=param['data_aug'],\
                   samples_max=12800, n_epoch=param['nb_epochs'], fixed_lambda=param['lambda'])

    print('\n\nFinal results:')
    print('Oracle model evaluation on clean data #%d:'%(test_data.size(0)))
    oracle_model.eval(x=test_data, y=test_label, batch_size=10)
    print('Substitute model evaluation on clean data: #%d:'%(test_data.size(0)))
    substitute_model.eval(x=test_data, y=test_label, batch_size=512)
    # test_adv = get_adv(model=substitute_model, x=test_data, y=test_label, epsilon=param['epsilon'])
    # print('Substitute model FGSM attack itself\'s accuracy on adversarial samples #%d:'%(test_adv.size(0)))
    # substitute_model.eval(x=test_adv, y=test_label, batch_size=512)
    # print('Oracle model FGSM attack\'s accuracy on adversarial samples #%d:'%(test_adv.size(0)))
    # oracle_model.eval(x=test_adv, y=test_label, batch_size=512)
