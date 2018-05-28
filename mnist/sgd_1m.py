#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 23:33:37 2018

@author: xueyunzhe
"""

import tensorflow as tf
import numpy as np
import time
import os


w_init = tf.contrib.layers.xavier_initializer()
w_regular = tf.contrib.layers.l2_regularizer(0.0002)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

batch_size=100


x = tf.placeholder(dtype=np.float32, shape=[batch_size, 200000])
y = tf.placeholder(dtype=np.int64, shape=[batch_size])
eta = tf.placeholder(dtype=np.float32)

mlp2 = tf.layers.dense(
    inputs=x,
    units=1000,
    activation=tf.identity,
    kernel_initializer=w_init,
    use_bias=True,
    trainable=True,
    name=None,
    reuse=None
    )

mlp2 = tf.layers.batch_normalization(
    inputs=mlp2,
    axis=-1,

)

mlp2 = tf.sigmoid(mlp2)

mlp = tf.layers.dense(
    inputs=mlp2,
    units=10,
    activation=tf.identity,
    kernel_initializer=w_init,
    use_bias=True,
    trainable=True,
    name=None,
    reuse=None
    )


loss_ = tf.losses.sparse_softmax_cross_entropy(
        labels=y, 
        logits=mlp,
        )
loss_re = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)#Loss of regularization
loss = tf.add_n([loss_]+loss_re)

accuracy = tf.contrib.metrics.accuracy(
        predictions=tf.argmax(mlp, 1),
        labels=y,
        )

optimizer = tf.train.MomentumOptimizer(learning_rate=eta, momentum=0.9, use_nesterov=True)

train_op = optimizer.minimize(loss)

def train(train_label, lr=0.005,monitor=None):
    train_loss = 0
    train_acc = 0
    train_data = np.load('train_data.npy', mmap_mode='r')
    n_iters = 50000//batch_size
    for i in range(n_iters):
        _, tmp_loss, tmp_acc = sess.run([train_op, loss, accuracy], 
                               feed_dict={
                                       x:train_data[i*batch_size:(i+1)*batch_size],
                                       y:train_label[i*batch_size:(i+1)*batch_size],
                                       eta: lr                                
       }
                               )
        
        train_loss += tmp_loss
        train_acc += tmp_acc
        
    train_loss /= n_iters
    train_acc /= n_iters
    if monitor:
        monitor['train_loss'] = train_loss
        monitor['train_acc'] = train_acc
    print('Training data\'s accuracy: %0.4f' %train_acc)
    print('Training data\'s loss: %0.4f'%train_loss)    
    
def test(test_label, monitor=None):
    test_loss = 0
    test_acc = 0
    test_data = np.load('test_data.npy', mmap_mode='r')
    n_iters = 10000//batch_size
    for i in range(n_iters):
        tmp_loss, tmp_acc = sess.run([loss, accuracy], 
                               feed_dict={
                                       x:test_data[i*batch_size:(i+1)*batch_size],
                                       y:test_label[i*batch_size:(i+1)*batch_size]
                                       }
                               )
        
        test_loss += tmp_loss
        test_acc += tmp_acc
        
    test_loss /= n_iters
    test_acc /= n_iters
    if monitor:
        monitor['test_loss'] = test_loss
        monitor['test_acc'] = test_acc        
    print('testing data\'s accuracy: %0.4f' %test_acc)
    print('testing data\'s loss: %0.4f'%test_loss)  
    

n_epoch = 240

(_, train_label), (_, test_label) = tf.keras.datasets.cifar10.load_data()

sess.run(tf.global_variables_initializer())

lr = 0.001
for i in range(1,n_epoch):
    if i==100 or i== 200:
        lr *= 0.1
    print('Epoch %d:'%i)
    train(train_label.flatten(), lr)
    test(test_label.flatten())

