#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 08/03/2018

import tensorflow as tf
import numpy as np

x_shape = [5, 3, 3, 2]
x = np.arange(reduce(lambda t, s: t*s, list(x_shape), 1))

print x

x = x.reshape([5, 3, 3, -1])

print x.shape



X = tf.Variable(x)

with tf.Session() as sess:
    m = tf.nn.moments(X, axes=[0])
    # m = tf.nn.moments(X, axes=[0,1])
    # m = tf.nn.moments(X, axes=np.arange(len(x_shape)-1))
    mean, variance = m

    print(sess.run(m, feed_dict={X: x}))
    # print(sess.run(m, feed_dict={X: x}))