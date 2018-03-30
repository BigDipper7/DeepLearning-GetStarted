#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 27/03/2018]

import tensorflow as tf
# from keras import backend as K
# from keras.applications.resnet50 import ResNet50


class Yggdrasil:
    in_width = 128
    in_height = 256
    in_channel = 3
    batch_size = 32
    keep_prob = 0.5
    epoch = 30

    def __init__(self, n_class):
        self.n_class = n_class

    def model(self, X):

        resnet50 = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=X, pooling=None)
        print("==== Shape : ====")
        print(resnet50.shape)
        print("==== Shape +++ ====")

        avg_pool = tf.nn.avg_pool(resnet50, ksize=[1, 1, 1, 1], strides=[1, 2, 2, 1], padding='SAME')

        flatten1 = tf.layers.flatten(avg_pool)  # Flatten the tensor and keep batch_size dim
        linear1 = tf.layers.dense(flatten1, units=512, activation=None, use_bias=True)

        bn1 = tf.layers.batch_normalization(linear1, axis=0)

        relu1 = tf.nn.relu(bn1)

        dropout1 = tf.nn.dropout(relu1, keep_prob=self.keep_prob)

        logits = tf.layers.dense(dropout1, units=self.n_class, activation=None, use_bias=True)

        return logits
