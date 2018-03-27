#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 27/03/2018]

import tensorflow as tf


class Yggdrasil:
    in_width = 80
    in_height = 180
    in_tunnel = 3
    batch_size = 32

    def __init__(self, n_class):
        self.n_class = n_class

        # X = tf.placeholder(tf.float32, shape=(None, self.in_height, self.in_width, self.in_tunnel))
        # Y = tf.placeholder(tf.float32, shape=(None, n_class))

        # tf.nn.softmax(l)

    def model(self, X):


        return None