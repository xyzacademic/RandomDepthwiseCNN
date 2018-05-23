import numpy as np
import os


train_list = []
test_list = []

for j in range(1,11):
    print(j)
    os.chdir('features%s'%str(j))
    for i in range(10):
        train_ = np.load('train_data_%d.npy'%i)
        train_list.append(train_)
        test_ = np.load('test_data_%d.npy'%i)
        test_list.append(test_)
    os.chdir(os.pardir)

train = np.concatenate(train_list, axis=1)
test = np.concatenate(test_list, axis=1)

np.save('train_data.npy', train)
np.save('test_data.npy', test)
