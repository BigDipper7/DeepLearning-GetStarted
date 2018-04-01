#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 07/01/2018

import numpy as np
import tensorflow as tf


x = np.arange(12)
print x

m = x.reshape([3, 2, 2])
print m

print m.reshape((1, 3, 2, 2))
print m.reshape((1, 2, 3, 2))

n = m.reshape((1, 3, 2, 2))
print '\n\n=======1=====\n', n

g = []
g.append(n)
g.append(n)
g.append(n)

print '\n\n=======2=====\n', g, np.asarray(g).shape


with tf.Session() as sess:
    print '\n\n=======3=====\n'
    r = sess.run(tf.concat(g, axis=0))
    print r, r.shape
