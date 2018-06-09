#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 20:39:27 2017

@author: wowjoy
"""


import tensorflow as tf
import numpy as np
import time
import sys
import os

k = int(sys.argv[1])
n = int(sys.argv[2])
fid = int(sys.argv[3])
foid = int(sys.argv[4])
gpu_ = str(sys.argv[5])

resume = True

model_name = '%d%d%d%d'%(k, n, fid, foid)

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_


w_init = tf.random_normal_initializer(mean=0, stddev=1)
b_init = tf.random_normal_initializer(mean=0, stddev=1)


def random_hp(data, kernel_size, num_layers, n):
    # num_layers = 6

    # kernel_size = 3

    conv = tf.contrib.layers.conv2d(
        inputs=data,
        num_outputs=n,
        kernel_size=kernel_size,
        stride=1,
        padding='VALID',
        activation_fn=tf.sign,
        weights_initializer=w_init,
        #        weights_regularizer=w_regular,
        biases_initializer=b_init,
        data_format="NCHW",
    )

    for i in range(num_layers - 1):
        var = tf.get_variable(str(i), [kernel_size, kernel_size, n, 1], initializer=w_init, dtype=tf.float32)
        conv = tf.nn.depthwise_conv2d(conv, var, [1, 1, 1, 1], 'VALID', data_format='NCHW')
        biases = tf.get_variable('bias_' + str(i), [n], initializer=b_init, dtype=tf.float32)
        conv = tf.nn.bias_add(conv, biases, data_format='NCHW')
        conv = tf.sign(conv)

    global_pool = tf.reduce_mean(input_tensor=conv, axis=[2,3])

    p = tf.contrib.layers.flatten(global_pool)

    # p = tf.sign(p)

    return p


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

x = tf.placeholder(tf.float32, shape=[None, 3, 96, 96])
# x_ = tf.transpose(x, [0, 3, 1, 2])


n_features = 2500
batch_size = 20

features = random_hp(data=x, kernel_size=k, num_layers=n, n=n_features)


def training(index, folderid):

    train_data = np.load('x_train.npy')
    test_data = np.load('x_test.npy')
    train_data = (train_data - train_data.mean(axis=(1, 2, 3), keepdims=True)) / train_data.std(axis=(1, 2, 3),
                                                                                                keepdims=True)
    test_data = (test_data - test_data.mean(axis=(1, 2, 3), keepdims=True)) / test_data.std(axis=(1, 2, 3),
                                                                                            keepdims=True)

    path = 'features%d/' % folderid
    train_features = np.zeros(shape=(train_data.shape[0], n_features), dtype=np.float32)
    for i in range(train_data.shape[0] // batch_size):
        tmp = train_data[i * batch_size:(i + 1) * batch_size]

        train_features[i * batch_size:(i + 1) * batch_size] = sess.run(features, feed_dict={x: tmp})
        if i % 100 == 0:
            print(i)

    np.save(path+'train_data_%d.npy' % index, train_features)
    del train_features

    test_features = np.zeros(shape=(test_data.shape[0], n_features), dtype=np.float32)
    for i in range(test_data.shape[0] // batch_size):
        tmp = test_data[i * batch_size:(i + 1) * batch_size]


        test_features[i * batch_size:(i + 1) * batch_size] = sess.run(features, feed_dict={x: tmp})
        # if i % 100 == 0:
        #     print(i)

    np.save(path+'test_data_%d.npy' % index, test_features)
    del test_features

def testing(index, folderid):
    test_data = np.load('adv/test_data_vgg1.npy')
    test_data = (test_data - test_data.mean(axis=(1, 2, 3), keepdims=True)) / test_data.std(axis=(1, 2, 3),
                                                                                            keepdims=True)

    path = 'features%d/'%(folderid+10)
    test_features = np.zeros(shape=(test_data.shape[0], n_features), dtype=np.float32)
    for i in range(test_data.shape[0] // batch_size):
        tmp = test_data[i * batch_size:(i + 1) * batch_size]

        test_features[i * batch_size:(i + 1) * batch_size] = sess.run(features, feed_dict={x: tmp})
        # if i % 100 == 0:
        #     print(i)

    np.save(path+'test_data_%d.npy' % (index+20), test_features)
    del test_features


    


a = time.time()
saver = tf.train.Saver()
if resume == False:
    sess.run(tf.global_variables_initializer())
    saver.save(sess=sess, save_path='logs2/%s/'%model_name, global_step=1)
    training(fid, foid)
elif resume == True:
    saver.restore(sess, 'logs2/%s/-1'%model_name)
    testing(fid, foid)


print('Cost: %.3f' % (time.time() - a))
