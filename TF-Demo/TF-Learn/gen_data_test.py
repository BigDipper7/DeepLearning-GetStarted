#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 31/12/2017

import numpy as np

NUM_SIZE = 5

X_real = np.linspace(-1, 1, num=NUM_SIZE)[:, np.newaxis]
print X_real
assert X_real.shape == (NUM_SIZE, 1)

X_noise = np.random.normal(0, 0.05, size=X_real.shape)  # 注意shape
print X_noise

Y_real = X_real ** 2 - 0.5 + X_noise
print X_real ** 2
print X_real ** 2 - 0.5
print Y_real

