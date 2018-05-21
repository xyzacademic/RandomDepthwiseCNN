
import tensorflow as tf
import numpy as np
import time
import sys
import os




k = int(sys.argv[1]) # kernel_size
n = int(sys.argv[2]) # the number of layers
fid = int(sys.argv[3]) # index contained in the name of features saved
foid = int(sys.argv[4]) # index of folder where the features will be saved to
gpu_ = str(sys.argv[5]) # device id


os.environ['CUDA_VISIBLE_DEVICES'] = gpu_


w_init = tf.random_normal_initializer(mean=0, stddev=1) # kernel initialization
b_init = tf.random_normal_initializer(mean=0, stddev=1) # bias initialization


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
        biases_initializer=b_init,
        data_format="NCHW",
    ) # typical convolutional kernel

    for i in range(num_layers - 1):
        var = tf.get_variable(str(i), [kernel_size, kernel_size, n, 1], initializer=w_init, dtype=tf.float32)
        conv = tf.nn.depthwise_conv2d(conv, var, [1, 1, 1, 1], 'VALID', data_format='NCHW') # depthwise convolution
        biases = tf.get_variable('bias_' + str(i), [n], initializer=b_init, dtype=tf.float32)
        conv = tf.nn.bias_add(conv, biases, data_format='NCHW')
        conv = tf.sign(conv)

    global_pool = tf.reduce_mean(input_tensor=conv, axis=[2,3]) # global average

    p = tf.contrib.layers.flatten(global_pool)

    return p


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

x = tf.placeholder(tf.float32, shape=[None, 3, 32, 32])

n_features = 2000 # the number of features you want to generate
batch_size = 500

features = random_hp(data=x, kernel_size=k, num_layers=n, n=n_features)


# load data
train_data = np.load('x_train.npy')
test_data = np.load('x_test.npy')

# normalization on each images
train_data = (train_data - train_data.mean(axis=(1, 2, 3), keepdims=True)) / train_data.std(axis=(1, 2, 3),
                                                                                                keepdims=True)
test_data = (test_data - test_data.mean(axis=(1, 2, 3), keepdims=True)) / test_data.std(axis=(1, 2, 3),
                                                                                            keepdims=True)

def save_features(index, folderid):
    path = 'features%d/'%folderid
    train_features = np.zeros(shape=(train_data.shape[0], n_features), dtype=np.float32)
    for i in range(train_data.shape[0] // batch_size):
        tmp = train_data[i * batch_size:(i + 1) * batch_size]
        train_features[i * batch_size:(i + 1) * batch_size] = sess.run(features, feed_dict={x: tmp})

    np.save(path+'train_data_%d.npy' % index, train_features)
    del train_features

    test_features = np.zeros(shape=(test_data.shape[0], n_features), dtype=np.float32)
    for i in range(test_data.shape[0] // batch_size):
        tmp = test_data[i * batch_size:(i + 1) * batch_size]
        test_features[i * batch_size:(i + 1) * batch_size] = sess.run(features, feed_dict={x: tmp})

    np.save(path+'test_data_%d.npy' % index, test_features)
    del test_features


a = time.time()

sess.run(tf.global_variables_initializer())
save_features(fid, foid)

print('Cost: %.3f' % (time.time() - a))