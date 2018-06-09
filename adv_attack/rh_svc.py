#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:09:32 2018

@author: xueyunzhe
"""

from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import accuracy_score
import time
import os
import tensorflow as tf
import pickle

resume = True



if resume == False:
    train_label = np.load('y_train.npy')
    test_label = np.load('y_test.npy')

    train = np.load('train_data.npy')
    test = np.load('test_data.npy')
    c = 0.5

    svc = LinearSVC(C=c, dual=False)
    a = time.time()
    svc.fit(X=train, y=train_label)
    print('Cost: %.3f seconds'%(time.time() - a))
    yp = svc.predict(test)

    print('Accuracy: ', accuracy_score(y_true=train_label, y_pred=svc.predict(train)))
    print('Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))

    with open('stl10_svc.pkl', 'wb') as f:
        pickle.dump(svc, f)


elif resume == True:
    with open('stl10_svc.pkl', 'rb') as f:
        svc = pickle.load(f)
    test = np.load('test_data6.npy')
    test_label = np.load('y_test.npy')
    yp = svc.predict(test)

    print('Accuracy: ', accuracy_score(y_true=test_label, y_pred=yp))
