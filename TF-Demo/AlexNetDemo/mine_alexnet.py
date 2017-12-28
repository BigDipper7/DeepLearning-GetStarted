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

# define global params
MODEL_PATH = './models'


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
weights = {                                                   # [None, 28, 28, 1]
    'c1': tf.Variable(tf.random_normal([11, 11, 1, 96])),     # [None, 28, 28, 96]
    'm1': 2,                                                  # [None, 14, 14, 96]
    'n1': 4,                                                  # [None, 14, 14, 96]
    'c2': tf.Variable(tf.random_normal([5, 5, 96, 256])),     # [None, 14, 14, 256]
    'm2': 2,                                                  # [None, 7, 7, 256]
    'n2': 4,                                                  # [None, 7, 7, 256]
    'c3': tf.Variable(tf.random_normal([3, 3, 256, 384])),    # [None, 7, 7, 384]
    'm3': 2,                                                  # [None, 4, 4, 384]
    'n3': 4,                                                  # [None, 4, 4, 384]
    'c4': tf.Variable(tf.random_normal([3, 3, 384, 384])),    # [None, 4, 4, 384]
    'c5': tf.Variable(tf.random_normal([3, 3, 384, 256])),    # [None, 4, 4, 256]
    'm5': 2,                                                  # [None, 4, 4, 256]
    'n5': 4,                                                  # [None, 4, 4, 256]
    'f1': tf.Variable(tf.random_normal([4*4*256, 4096])),     # [None, 4096]
    'f2': tf.Variable(tf.random_normal([4096, 4096])),        # [None, 4096]
    'out': tf.Variable(tf.random_normal([4096, 10])),         # [None, 10]                                                     # [None, 4, 4, 256]
}

strides = {
    'c1': 1,
    'm1': 2,
    'c2': 1,
    'm2': 2,
    'c3': 1,
    'm3': 2,
    'c4': 1,
    'c5': 1,
    'm5': 2,
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
def alexnet(X, weights, bias, strides):
    X = tf.reshape(X, [-1, 28, 28, 1], name='inputX')
    # [None, 28, 28, 1]

    conv1 = conv2d(X, weights['c1'], strides['c1'], bias['c1'], name='conv1')
    # [None, 28, 28, 96]
    maxpool1 = maxpooling2d(conv1, weights['m1'], strides['m1'], name='maxpool1')
    # [None, 14, 14, 96]
    norm1 = norm4d(maxpool1, 5, name='norm1')
    # [None, 14, 14, 96]

    conv2 = conv2d(norm1, weights['c2'], strides['c2'], bias['c2'], name='conv2')
    # [None, 14, 14, 256]
    maxpool2 = maxpooling2d(conv2, weights['m2'], strides['m2'], name='maxpool2')
    # [None, 7, 7, 256]
    norm2 = norm4d(maxpool2, 5, name='norm2')
    # [None, 7, 7, 256]

    conv3 = conv2d(norm2, weights['c3'], strides['c3'], bias['c3'], name='conv3')
    # [None, 7, 7, 384]
    maxpool3 = maxpooling2d(conv3, weights['m3'], stride=strides['m3'], name='maxpool3')
    # [None, 4, 4, 384]
    norm3 = norm4d(maxpool3, 5, name='norm3')
    # [None, 4, 4, 384]

    conv4 = conv2d(norm3, weights['c4'], strides['c4'], bias['c4'], name='conv4')
    # [None, 4, 4, 384]
    conv5 = conv2d(conv4, weights['c5'], strides['c5'], bias['c5'], name='conv5')
    # [None, 4, 4, 384]
    maxpool5 = maxpooling2d(conv5, weights['m5'], strides['m5'])
    norm5 = norm4d(maxpool5, 5, name='norm5')
    # [None, 2, 2, 256]

    fc1 = tf.reshape(norm5, [-1, weights['f1'].get_shape().as_list()[0]], name='reshape_to_vector')
    # [None, 4 * 4 * 256]
    fc1 = tf.matmul(fc1, weights['f1'])
    # [None, 4096]
    fc1 = tf.nn.bias_add(fc1, bias['f1'])
    fc1 = tf.nn.relu(fc1, name='relu_fc1')
    # [None, 4096]

    # dropout
    drop1 = tf.nn.dropout(fc1, keep_prob=dropout)
    # [None, 4096]

    fc2 = tf.nn.bias_add(tf.matmul(tf.reshape(drop1, [-1, weights['f2'].get_shape().as_list()[0]]), weights['f2']), bias['f2'])
    fc2 = tf.nn.relu(fc2)
    # [None, 4096]

    # dropout
    drop2 = tf.nn.dropout(fc2, keep_prob=dropout)
    # [None, 4096]

    out = tf.nn.bias_add(tf.matmul(drop2, weights['f2']), bias['out'])
    # [None, 10]

    return out


with tf.Session() as sess:
    import os
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
        print 'mkdir {%s}' % MODEL_PATH
    global_step = tf.Variable(0, trainable=False)
    saver = tf.train.Saver(max_to_keep=10)
    print 'prepared global_sept setting finished....'

    init = tf.global_variables_initializer()
    x = tf.placeholder(dtype='float32', shape=[None, n_input_size], name='oriX')
    y = tf.placeholder(dtype='float32', shape=[None, n_output_classes_size], name='oriY')

    # define network
    pred = alexnet(x, weights, bias, strides)
    # calculate loss
    loss = tf.reduce_mean(y - pred)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    print 'All Definitions are prepared....'

    print 'Begin to prepare initializer'
    sess.run(init)
    print 'Initializer preparation finished....'
    for x in range(0, iter_len):
