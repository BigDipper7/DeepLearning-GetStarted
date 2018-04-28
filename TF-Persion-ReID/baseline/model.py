#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 27/03/2018]

import tensorflow as tf
from utils.const import IN_HEIGHT, IN_WIDTH, IN_CHANNEL, BATCH_SIZE, EPOCH
# from keras import backend as K
# from keras.applications.resnet50 import ResNet50


class Yggdrasil:
    in_width = IN_WIDTH
    in_height = IN_HEIGHT
    in_channel = IN_CHANNEL
    batch_size = BATCH_SIZE
    keep_prob = 0.5
    epoch = EPOCH
    n_class = 1
    n_dataset_len = 1
    n_invoke_step_per_epoch = 50

    def __init__(self, n_class, n_dataset_len):
        Yggdrasil.n_class = n_class
        Yggdrasil.n_dataset_len = n_dataset_len
        # self.resnet = None

    def extract_features(self, X):
        resnet50 = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=X, pooling=None)
        print("==== Shape : ====")
        print(resnet50.output_shape)
        print("==== Shape +++ ====")
        return resnet50

    def model(self, X):

        with tf.name_scope('network'):
            with tf.name_scope('resnet50'):
                # resnet50 = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=X, pooling=None)
                resnet50 = self.extract_features(X)
                # self.resnet = resnet50

            with tf.name_scope('avg_pool1'):
                # 注意的是resnet50本身不是tensor，所以需要指定输出的tensor是什么
                avg_pool = tf.nn.avg_pool(resnet50.output, ksize=[1, 1, 1, 1], strides=[1, 2, 2, 1], padding='SAME')

            with tf.name_scope('fully_connected1'):
                flatten1 = tf.layers.flatten(avg_pool)  # Flatten the tensor and keep batch_size dim
                linear1 = tf.layers.dense(flatten1, units=512, activation=None, use_bias=True)

                bn1 = tf.layers.batch_normalization(linear1, axis=-1)

                relu1 = tf.nn.relu(bn1)

                dropout1 = tf.nn.dropout(relu1, keep_prob=self.keep_prob)

            with tf.name_scope('logits'):
                logits = tf.layers.dense(dropout1, units=self.n_class, activation=None, use_bias=True)

        return logits
