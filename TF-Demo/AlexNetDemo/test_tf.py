#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 28/12/2017

import tensorflow as tf
import numpy as np

x = []
for i in range(0, 20):
    x += [i]

print x

# trans to float32
x1 = np.asarray(x, dtype=np.float32)
print 'new x:'
print x1

with tf.Session() as sess:
    m = np.reshape(x, [-1, 5])
    print 'int m: [%s]' % (str(m.shape))
    print m
    print sess.run(tf.reduce_mean(m))
    print sess.run(tf.reduce_mean(m, axis=0))
    print sess.run(tf.reduce_mean(m, axis=1))

    m = np.reshape(x1, [-1, 5])
    print 'float m: [%s]' % (str(m.shape))
    print m
    print sess.run(tf.reduce_mean(m))
    print sess.run(tf.reduce_mean(m, axis=0))
    print sess.run(tf.reduce_mean(m, axis=1))

