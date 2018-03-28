#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 12/03/2018

import tensorflow as tf
from keras import backend as K
import numpy as np

from model import Yggdrasil
from utils.util import curr_normal_time

# define super-params
n_classes = 1000


yggdrasil = Yggdrasil(n_class=n_classes)


with tf.Session() as sess:

    global_step = tf.Variable(0, trainable=False)

    X = tf.placeholder(tf.float32, shape=(None, Yggdrasil.in_height, Yggdrasil.in_width, Yggdrasil.in_tunnel))
    Y = tf.placeholder(tf.float32, shape=(None, yggdrasil.n_class))

    logits = yggdrasil.model(X)

    inference = tf.nn.softmax(logits=logits)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))

    # decay=5e-4,
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, use_nesterov=True).\
        minimize(loss, global_step=global_step,)

    # initialize
    init = tf.global_variables_initializer()
    sess.run(init)

    print("======== Training Begin ========")
    while global_step.eval() < Yggdrasil.epoch:
        _, cal_loss = sess.run([optimizer, loss], feed_dict={X: None, Y: None})
        print("%s : epoch: [%d] with loss [%.8f]" % (curr_normal_time(), global_step.eval(), cal_loss))

    print("======== Training Finished ========")

    print("======== Testing Begin ========")
    test_X = None
    test_Y = None
    prediction = sess.run([inference], feed_dict={X: test_X})
    predict_label = np.argmax(prediction, axis=-1)
    ground_truth_label = np.argmax(test_Y, axis=-1)
    correct_num = 0
    total_num = len(ground_truth_label)
    for i in xrange(total_num):
        if predict_label[i] == ground_truth_label[i]:
            correct_num += 1
            # print()
    print("Acc: %.3f" % (float(correct_num)/total_num))







