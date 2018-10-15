import torch
import torchvision
import torchvision.transforms as transforms
from random_hp import Rh
import sys
import torch.backends.cudnn as cudnn
import argparse
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import accuracy_score
import time
import os

###
# Global Flag
###

parser = argparse.ArgumentParser(description='MNIST features generating')

parser.add_argument('--kernel-size', default=3, type=int, help='Size of convolutional kernel filter.')
parser.add_argument('--layers', default=2, type=int, help='The number of block layers')
# parser.add_argument('--folder-id', default=0, type=int, help='Index of folder in which features save.')
# parser.add_argument('--fid', default=0, type=int, help='Index of file which features save as.')
parser.add_argument('--fp16', action='store_true', help='Run model fp16 mode.')
parser.add_argument('--resume', action='store_true', help='Run model fp16 mode.')
parser.add_argument('--num-features', default=1000, type=int, help='The number of features.')
parser.add_argument('--batch-size', default=2, type=int, help='Batch size.')
parser.add_argument('--seed', default=2018, type=int, help='Random seed.')
parser.add_argument('--gpu', default=-1, type=int, help='Using which gpu.')
args = parser.parse_args()

def get_features():
    DATA = 'mnist'

    if DATA == 'mnist':
        train_dir = 'mnist'
        test_dir = 'mnist'



    use_cuda = True

    dtype = torch.float16 if args.fp16 else torch.float32
    k = args.kernel_size
    n = args.layers
    save_path = 'checkpoints_%d_%d' % (k, n)
    gpu = args.gpu
    n_features = args.num_features
    batch_size = args.batch_size

    seed = args.seed
    print('Random seed: ', seed)
    torch.manual_seed(seed)


    net = Rh(n, k, n_features)

    test_transform = transforms.Compose(
            [
                    transforms.ToTensor(),
                    ])

    trainset = torchvision.datasets.MNIST(root=train_dir, train=True, download=True, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4,
                                               pin_memory=True)
    testset = torchvision.datasets.MNIST(root=test_dir, train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4,
                                              pin_memory=True)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(save_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(save_path + '/ckpt_%d.t7'%n_features)
        net.load_state_dict(checkpoint['net'])

    else:
        print('Saving...')
        state = {
            'net': net.state_dict(),
        }

        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        torch.save(state, save_path + '/ckpt_%d.t7'%n_features)

    if use_cuda:
        print('start move to cuda')
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True
        if args.fp16:
            print('Using float point 16')
            net = net.half()
        if gpu == -1:
            net = torch.nn.DataParallel(net, device_ids=[0,1])
            gpu = 0
        device = torch.device("cuda:%d"%gpu)
        net.to(device=device)


    def save_features():
        net.eval()

        with torch.no_grad():
            train_features = np.zeros(shape=(60000, n_features), dtype=np.float16)
            train_label = np.zeros(shape=(60000,), dtype=np.int16)

            for batch_idx, (data, target) in enumerate(train_loader):
                if use_cuda:
                    data, target = data.to(device=device, dtype=dtype), target.to(device=device)

                outputs = net(data)
                train_features[batch_idx * batch_size:(batch_idx + 1) * batch_size] = outputs.data.cpu()
                train_label[batch_idx * batch_size:(batch_idx + 1) * batch_size] = target.data.cpu()

            test_features = np.zeros(shape=(10000, n_features), dtype=np.float16)
            test_label = np.zeros(shape=(10000,), dtype=np.int16)



            for batch_idx, (data, target) in enumerate(test_loader):
                if use_cuda:
                    data, target = data.to(device=device, dtype=dtype), target.to(device=device)
                outputs = net(data)
                test_features[batch_idx * batch_size:(batch_idx + 1) * batch_size] = outputs.data.cpu()
                test_label[batch_idx * batch_size:(batch_idx + 1) * batch_size] = target.data.cpu()



        return train_features, train_label, test_features, test_label

    return save_features()


a = time.time()
train_data, train_label, test_data, test_label = get_features()
print('get data successfully')
print('Cost: %.3f' % (time.time() - a))
a = time.time()
if not os.path.isdir('temp_data'):
    os.mkdir('temp_data')
os.chdir('temp_data')
np.save('train_data_%d_%d_%d.npy' % (args.kernel_size, args.layers, args.num_features), train_data)
np.save('train_label_%d_%d_%d.npy' % (args.kernel_size, args.layers, args.num_features), train_label)
np.save('test_data_%d_%d_%d.npy' % (args.kernel_size, args.layers, args.num_features), test_data)
np.save('test_label_%d_%d_%d.npy' % (args.kernel_size, args.layers, args.num_features), test_label)
os.chdir(os.pardir)
print('save data successfully')
# print('Cost: %.3f' % (time.time() - a))
# print('training data size: ')
# print(train_data.shape)
# print('testing data size: ')
# print(test_data.shape)
# c = 0.005
#
# train = train_data
# test = test_data
#
# svc = LinearSVC(C=c, dual=False)
# a = time.time()
# svc.fit(X=train, y=train_label)
# print('Cost: %.3f seconds'%(time.time() - a))
# yp = svc.predict(test)
#
# print('Accuracy: ', accuracy_score(y_true=train_label, y_pred=svc.predict(train)))
# print('Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))