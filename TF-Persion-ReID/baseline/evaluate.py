#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 19/04/2018

import tensorflow as tf
import numpy as np
from model import Yggdrasil
from utils.datasets import get_ds_iterators
from utils.const import LOG_DIR


n_classes_train, train_os_iterator, valid_os_iterator = get_ds_iterators()

ygg = Yggdrasil(n_classes_train, 10000)


with tf.name_scope('_sess'):
    with tf.name_scope('_dataset'):
        next_op_train = train_os_iterator.get_next()
        next_op_valid = valid_os_iterator.get_next()

    with tf.name_scope('_init'):
        init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    with tf.name_scope('saver_ops'):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(LOG_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore model')
            saver.restore(sess, ckpt.model_checkpoint_path)