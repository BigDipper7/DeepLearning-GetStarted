#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 01/01/2018

import tensorflow as tf
import numpy as np

from utils import util


NUM_X_SIZE = 300

X_real = np.linspace(-1, 1, num=NUM_X_SIZE)[:, np.newaxis]
assert X_real.shape == (NUM_X_SIZE, 1)

X_noise = np.random.normal(0, 0.05, size=X_real.shape)
assert X_real.shape == X_noise.shape

# y = x^2 - 0.5
Y_real = X_real ** 2 - 0.5 + X_noise

x = tf.placeholder(dtype=tf.float32, shape=[NUM_X_SIZE, 1], name='inputX')
y = tf.placeholder(dtype=tf.float32, shape=[NUM_X_SIZE, 1], name='outputY')
batch_size = 30
iter_size = 2000

all_data = {'X': X_real, 'Y': Y_real}

weights = {
    'h1': tf.Variable(tf.random_normal([1, 20], dtype=tf.float32)),
    'h2': tf.Variable(tf.random_normal([20, 30], dtype=tf.float32)),
    'out': tf.Variable(tf.random_normal([30, 1], dtype=tf.float32)),
}

bias = {
    'h1': tf.Variable(tf.random_normal([20], dtype=tf.float32)),
    'h2': tf.Variable(tf.random_normal([30], dtype=tf.float32)),
    'out': tf.Variable(tf.random_normal([1], dtype=tf.float32)),
}


def hidden_op(X, weights, bias, name='hidden'):
    X_mul = tf.matmul(X, weights)
    X_mul_plus_b = tf.nn.bias_add(X_mul, bias)
    X_relu = tf.nn.relu(X_mul_plus_b, name=name)
    return X_relu


def networks(X, weights, bias):
    h1 = hidden_op(X, weights['h1'], bias['h1'], name='h1')
    h2 = hidden_op(h1, weights['h2'], bias['h2'], name='h2')

    out = tf.nn.bias_add(tf.matmul(h2, weights['out']), bias['out'], name='out')

    return out


with tf.Session() as sess:
    global_step = tf.Variable(0, trainable=False, dtype=tf.int32)

    # Train
    logits = networks(x, weights, bias)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y-logits), axis=1))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss, global_step=global_step)

    # Test
    test = networks(x, weights, bias)
    # accuracy = tf.

    # initializer
    init = tf.global_variables_initializer()
    sess.run(init)

    while global_step.eval() < iter_size:
        sess.run(optimizer, feed_dict={x: all_data['X'], y: all_data['Y']})

        # sess.run(global_step.assign_add(1)) # 这句话就不用了，因为你前面在损失函数的时候设定过global_step了
        # print global_step.eval()

        if global_step.eval() % 50 == 0:
            start = util.curr_timestamp_time()
            los = sess.run(loss, feed_dict={x: all_data['X'], y: all_data['Y']})
            time_span = util.time_span(start)
            print '[%d] ** current cost time: %.5f, loss: %.5f' % (global_step.eval(), time_span, los)

    print 'Finished !'

    start = util.curr_timestamp_time()
    los = sess.run(loss, feed_dict={x: all_data['X'], y: all_data['Y']})
    time_span = util.time_span(start)
    print 'FINAL *** loss: %.5f, cost time: %.5f' % (los, time_span)



