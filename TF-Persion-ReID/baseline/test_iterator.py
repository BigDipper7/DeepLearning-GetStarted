#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 01/04/2018


import tensorflow as tf
from utils.const import DS_ROOT_PTH
from utils.datasets import get_data_list


def _map_func(dic_item):
    im = tf.read_file(dic_item['images'])
    im = tf.image.decode_jpeg(im)
    im = tf.image.resize_image_with_crop_or_pad(im, 224, 224)
    return {'image': im, 'label': dic_item['labels']}

_dataset_train, _dataset_test = get_data_list(DS_ROOT_PTH)

ds = tf.data.Dataset.from_tensor_slices(_dataset_train)
ds = ds.map(_map_func)
iterator = ds.make_one_shot_iterator()

next_ele = iterator.get_next()

with tf.Session() as sess:

    X = tf.placeholder(dtype=tf.float32, shape=(224, 224, 3))
    net = tf.layers.dense(X, 200, None)

    sess.run(tf.global_variables_initializer())

    for i in range(2000):
        # l = sess.run(next_ele)
        l = sess.run(iterator.get_next()) # 这种写法不推荐，应该使用下面这种写法
        #UserWarning: An unusually high number of `Iterator.get_next()` calls was detected. This often indicates that `Iterator.get_next()` is being called inside a training loop, which will cause gradual slowdown and eventual resource exhaustion. If this is the case, restructure your code to call `next_element = iterator.get_next() once outside the loop, and use `next_element` inside the loop.
        # warnings.warn(GET_NEXT_CALL_WARNING_MESSAGE)

        m= sess.run([net], feed_dict={X: l['image']})
        print(i, len(m))