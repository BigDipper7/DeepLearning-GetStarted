#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 12/10/2017

import numpy as np


x = np.arange(0, 12)

print " ============= "
print x.shape
print " ------ "
print x

# resize [12, 0] to [2, 2, 3]
x.resize(2, 2, 3)

print " ============= "
print x.shape
print " ------ "
print x

print "\n\n ====== 1 ======= \n\n"

# get default transpose func
print x.transpose()
print "\n\n ------ \n\n"
print x.transpose((2, 1, 0))
# this is equals: x.transpose() == x.transpose(2, 1, 0)


# Also!
print "\n\n ====== 2 ======= \n\n"
print x.transpose((2, 0, 1))
print "\n\n ------ \n\n"
print x.swapaxes(0, 1).swapaxes(0, 2)
# this also means x.transpose() == x.swapaxes(axis1, axis2)[multi]
# 高纬度的矩阵转置 等同于 多维度的轴的交换，但是对于某种情况下我们考虑处理数据会更加的方便



