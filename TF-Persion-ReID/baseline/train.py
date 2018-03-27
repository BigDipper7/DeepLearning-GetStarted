#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 12/03/2018

import tensorflow as tf
import numpy as np



with tf.Session() as sess:
    global_step = tf.Variable(0, trainable=False)


    init = tf.global_variables_initializer()
    sess.run(init)


