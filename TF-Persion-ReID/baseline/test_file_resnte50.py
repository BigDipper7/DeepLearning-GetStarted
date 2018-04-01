#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 31/03/2018

import tensorflow as tf
import numpy as np


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    X = tf.placeholder(dtype=tf.float32, shape=(None, 197,197,3)) # 注意最小的input shape 不能小于197
    resnet50 = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=X)
    m = tf.layers.flatten(resnet50.output)
    print(resnet50.output.shape)

    m = sess.run(m, feed_dict={X: np.random.random((77, 197, 197, 3))})
    print(m.shape)