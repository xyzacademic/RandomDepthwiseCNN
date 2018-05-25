

from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import accuracy_score
import time
import os
import tensorflow as tf


train = np.load('train_data.npy')
test = np.load('test_data.npy')
train_label = np.load('y_train.npy')
test_label = np.load('y_test.npy')


c = 0.01

svc = LinearSVC(C=c, dual=False)
a = time.time()
svc.fit(X=train, y=train_label)
print('Cost: %.3f seconds'%(time.time() - a))
yp = svc.predict(test)

print('Accuracy: ', accuracy_score(y_true=train_label,y_pred=svc.predict(train)))
print('Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))


