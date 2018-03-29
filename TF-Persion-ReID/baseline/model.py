#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 27/03/2018]

import tensorflow as tf
import os
# from keras import backend as K
# from keras.applications.resnet50 import ResNet50


class Yggdrasil:
    in_width = 80
    in_height = 180
    in_tunnel = 3
    batch_size = 32
    keep_prob = 0.5
    epoch = 30

    def __init__(self, n_class):
        self.n_class = n_class

    def model(self, X):

        resnet50 = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=X)
        print(resnet50.shape)

        avg_pool = tf.nn.avg_pool(resnet50, ksize=[1, 1, 1, 1], strides=[1, 2, 2, 1], padding='SAME')

        flatten1 = tf.layers.flatten(avg_pool)  # Flatten the tensor and keep batch_size dim
        linear1 = tf.layers.dense(flatten1, units=512, activation=None, use_bias=True)

        bn1 = tf.layers.batch_normalization(linear1, axis=0)

        relu1 = tf.nn.relu(bn1)

        dropout1 = tf.nn.dropout(relu1, keep_prob=self.keep_prob)

        logits = tf.layers.dense(dropout1, units=self.n_class, activation=None, use_bias=True)

        return logits

    def get_data_list(self, root_pth):
        _dataset_train = {'images':[], 'labels':[]}

        train_pth = os.path.join(root_pth, "bounding_box_train")
        # val_pth = os.path.join(root_pth, "val")
        test_pth = os.path.join(root_pth, "bounding_box_test")

        for root, dirs, files in os.walk(train_pth, topdown=True):
            print(root)
            print(dirs)
            print(files)
            for file_name in files:
                if not file_name[-4:] == '.jpg':
                    continue
                fsplits = file_name.split('_')
                tmp_label = fsplits[0]
                _dataset_train['images'].append(os.path.join(root, file_name))
                _dataset_train['labels'].append(os.path.join(tmp_label))

        return _dataset_train


