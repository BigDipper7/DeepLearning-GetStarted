#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 19/04/2018

import tensorflow as tf
import numpy as np
from tensorflow.python.framework.errors_impl import OutOfRangeError

from model import Yggdrasil
from utils.datasets import get_ds_iterators
from utils.const import LOG_DIR


n_classes_train, train_os_iterator, valid_os_iterator = get_ds_iterators()

ygg = Yggdrasil(n_classes_train, 10000)


with tf.name_scope('sess'):
    with tf.name_scope('_placeholders'):
        X = tf.placeholder(tf.float32, shape=(None, Yggdrasil.in_height, Yggdrasil.in_width, Yggdrasil.in_channel), name="X")
        Y = tf.placeholder(tf.float32, shape=(None, Yggdrasil.n_class), name="Y")

    with tf.name_scope('_dataset'):
        next_op_train = train_os_iterator.get_next()
        next_op_valid = valid_os_iterator.get_next()

    with tf.name_scope('_extract_features'):
        get_features = ygg.extract_features(X=X)
        features = get_features.output

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

    while True:
        try:
            next_valid_ds = sess.run(next_op_valid)
            fs = sess.run(tf.squeeze(features), feed_dict={X: next_valid_ds['image']})
            lbs = next_valid_ds['label']
            cids = next_valid_ds['cid']
            pids = next_valid_ds['pid']
            print(fs, fs.shape)
            print(lbs, lbs.shape)
            print(cids, cids.shape)
            print(pids, pids.shape)
            print("--------------------------------------------")

        except OutOfRangeError:
            print("Finish Iteration...")
            break

