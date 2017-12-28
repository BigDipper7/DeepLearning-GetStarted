#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 14/12/2017

import tensorflow as tf


# load data model
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./tmp/data/train", one_hot=True)


# define network hyper-params
learning_rate = 0.001
training_iter = 200000
batch_size = 128
display_step = 10


# set network parameters
n_input = 748  # 28 x 28 image shape
n_classes = 10  # 10 classes, count from 1 to 10
n_dropout = 0.75  # 0.75 probability to keep input

# set placeholders
X = tf.placeholder(dtype='float32', shape=[None, n_input], name='inputX')
y = tf.placeholder(dtype='float32', shape=[None, 10], name='resultY')
keep_prob = tf.placeholder(dtype='float32')

