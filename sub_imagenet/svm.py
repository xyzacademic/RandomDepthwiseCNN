from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import accuracy_score
import time
import os
import sys
import pickle

mid_fix = '_'+sys.argv[1] if sys.argv[0] is not '0' else ''




os.chdir('temp_data')
train_data = np.load('train_data%s_100000.npy'%mid_fix)
test_data = np.load('test_data%s_100000.npy'%mid_fix)
train_label = np.load('train_label%s_100000.npy'%mid_fix)
test_label = np.load('test_label%s_100000.npy'%mid_fix)
os.chdir(os.pardir)
print('training data size: ')
print(train_data.shape)
print('testing data size: ')
print(test_data.shape)
c = 0.005

train = train_data
test = test_data

svc = LinearSVC(C=c, dual=False)
a = time.time()
svc.fit(X=train, y=train_label)
print('Cost: %.3f seconds'%(time.time() - a))
yp = svc.predict(test)

print('Accuracy: ', accuracy_score(y_true=train_label, y_pred=svc.predict(train)))
print('Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))

save_path = 'checkpoints_svm%s'%mid_fix
if not os.path.isdir(save_path):
    os.mkdir(save_path)

with open(os.path.join(save_path, 'svm.pkl'), 'wb') as f:
    pickle.dump(svc, f)
