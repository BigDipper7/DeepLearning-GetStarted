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


def norm4d(X, radius, name):
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
    'out': tf.Variable(dtype='float32', expected_shape=[4096, 10]),         # [None, 10]                                                      # [None, 4, 4, 256]
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
