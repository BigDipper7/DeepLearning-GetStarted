#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 15/12/2017

import tensorflow as tf

# prepare data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
train = mnist.train
valid = mnist.validation
test = mnist.test

# define hyper-params for network
learning_rate = 0.001
iter_len = 20000
batch_size = 128
dropout = 0.75

# define network
n_input_size = 28 * 28
n_output_classes_size = 10


# define tf graph func
def conv2d(X, fil, stride, b, padding='SAME', name='conv_default'):
    conv = tf.nn.conv2d(input=X, filter=fil, strides=[1, stride, stride, 1], padding=padding, name=name)
    x_b = tf.nn.bias_add(conv, bias=b)
    y = tf.nn.relu(x_b)
    return y


def maxpooling2d(X, k, stride, padding='SAME', name='maxpool_default'):
    maxpool = tf.nn.max_pool(X, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding=padding, name=name)
    return maxpool


def norm4d(X, radius, name='norm_default'):
    norm = tf.nn.lrn(X, depth_radius=radius, bias=1.0, alpha=0.001/9.0, beta=0.75, name=name)
    return norm


# define AlexNet Params
weights = {                                                                 # [None, 28, 28, 1]
    'c1': tf.Variable(dtype='float32', expected_shape=[11, 11, 1, 96]),     # [None, 28, 28, 96]
    'c2': tf.Variable(dtype='float32', expected_shape=[5, 5, 96, 256]),     # [None, 28, 28, 256]
    'm1': 2,                                                                # [None, 14, 14, 256]
    'c3': tf.Variable(dtype='float32', expected_shape=[3, 3, 256, 384]),    # [None, 14, 14, 384]
    'm2': 2,                                                                # [None, 7, 7, 384]
    'c4': tf.Variable(dtype='float32', expected_shape=[3, 3, 384, 384]),    # [None, 7, 7, 384]
    'c5': tf.Variable(dtype='float32', expected_shape=[3, 3, 384, 256]),    # [None, 7, 7, 256]
    'm3': 2,                                                                # [None, 4, 4, 256]
    'f1': tf.Variable(dtype='float32', expected_shape=[4*4*256, 4096]),     # [None, 4096]
    'f2': tf.Variable(dtype='float32', expected_shape=[4096, 4096]),        # [None, 4096]
    'out': tf.Variable(dtype='float32', expected_shape=[4096, 10]),         # [None, 10]                                                     # [None, 4, 4, 256]
}

strides = {
    'c1': 4,
    'c2': 1,
    'm1': 2,
    'c3': 1,
    'm2': 2,
    'c4': 1,
    'c5': 1,
    'm3': 2,
}

bias = {                                                        # [None, 28, 28, 1]
    'c1': tf.Variable(dtype='float32', expected_shape=[96]),    # [None, 28, 28, 96]
    'c2': tf.Variable(dtype='float32', expected_shape=[256]),   # [None, 28, 28, 256]
    'c3': tf.Variable(dtype='float32', expected_shape=[384]),   # [None, 14, 14, 384]
    'c4': tf.Variable(dtype='float32', expected_shape=[384]),   # [None, 7, 7, 384]
    'c5': tf.Variable(dtype='float32', expected_shape=[256]),   # [None, 7, 7, 256]
    'f1': tf.Variable(dtype='float32', expected_shape=[4096]),  # [None, 4096]
    'f2': tf.Variable(dtype='float32', expected_shape=[4096]),  # [None, 4096]
    'out': tf.Variable(dtype='float32', expected_shape=[10]),   # [None, 10]
}

# define network
def alexnet(X):
    X = tf.reshape(X, [-1, 28, 28, 1], name='inputX')
    # [None, 28, 28, 1]

    conv1 = conv2d(X, weights['c1'], strides['c1'], bias['c1'], name='conv1')
    # [None, 28, 28, 96]
    norm1 = norm4d(conv1, 5, name='norm1')
    # [None, 28, 28, 96]

    conv2 = conv2d(norm1, weights['c2'], strides['c2'], bias['c2'], name='conv2')
    # [None, 28, 28, 256]
    norm2 = norm4d(conv2, 5, name='norm2')
    # [None, 28, 28, 256]

    maxpool1 = maxpooling2d(norm2, weights['m1'], strides['m1'], name='maxpool1')
    # [None, 14, 14, 256]

    conv3 = conv2d(maxpool1, weights['c3'], strides['c3'], bias['c3'], name='conv3')
    # [None, 14, 14, 384]
    norm3 = norm4d(conv3, 5, name='norm3')
    # [None, 14, 14, 384]

    maxpool2 = maxpooling2d(norm3, weights['m2'], strides['m2'], name='maxpool2')
    # [None, 7, 7, 384]

    conv4 = conv2d(maxpool2, weights['c4'], strides['c4'], bias['c4'], name='conv4')
    # [None, 7, 7, 384]
    norm4 = norm4d(conv4, 5, name='norm4')
    # [None, 7, 7, 384]

    conv5 = conv2d(norm4, weights['c5'], strides['c5'], bias['c5'], name='conv5')
    # [None, 7, 7, 256]
    norm5 = norm4d(conv5, 5, name='norm5')
    # [None, 7, 7, 256]

    maxpool3 = maxpooling2d(norm5, weights['m3'], strides['m3'], name='maxpool3')
    # [None, 4, 4, 256]

    fc1 = tf.reshape(maxpool3, [-1, 4 * 4 * 256], name='reshape_to_vector')
    # [None, 4 * 4 * 256]
    fc1 = tf.matmul(fc1, weights['f1'])
    # [None, 4096]
    fc1 = tf.nn.bias_add(fc1, bias['f1'])
    fc1 = tf.nn.relu(fc1, name='relu_fc1')

    fc2 = tf.nn.bias_add(tf.matmul(fc1, weights['f2']), bias['f2'])
    fc2 = tf.nn.relu(fc2)

    out = tf.nn.bias_add(tf.matmul(fc2, weights['f2']), bias['out'])

    return out

