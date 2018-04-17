#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 15/04/2018

import tensorflow as tf

ipt = tf.keras.layers.Input(shape=(197, 224, 3))
resnet = tf.keras.applications.ResNet50(False, 'imagenet', input_tensor=ipt)

for layer in resnet.layers:
    print('-----------------------------------------------------------------------')
    print(layer)
    print(layer.input)
    print(layer.output)