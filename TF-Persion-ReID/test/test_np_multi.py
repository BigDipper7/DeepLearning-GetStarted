#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 04/01/2018

import numpy as np


x = np.arange(1, 13)
print 'X:', x

m = x.reshape((2, 2, 3))
print 'm:', m

m2 = m.reshape((2, 2, 3, 1, 1))
print 'm2:', m2

m1s = np.ones((2, 2, 3, 5, 5))
print 'm1s', m1s

mRes = np.multiply(m2, m1s)
print 'mRes:', mRes
# 说明了这种乘法其实相当于直接的扩展。。。没什么特殊性

mRes2 = np.multiply(m, m1s)
print 'mRes2:', mRes2
# 上面这段话是根本就不能运行的：
# Traceback (most recent call last):
#   File "/Users/violinsolo/GitHub/DeepLearning-GetStarted/TF-Persion-ReID/test/test_np_multi.py", line 25, in <module>
#     mRes2 = np.multiply(m, m1s)
# ValueError: operands could not be broadcast together with shapes (2,2,3) (2,2,3,5,5)
