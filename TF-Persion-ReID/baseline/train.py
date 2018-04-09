#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 12/03/2018

import tensorflow as tf
from keras import backend as K
import numpy as np
import os

from model import Yggdrasil
from utils.util import curr_normal_time, cal_acc
from utils.datasets import get_data_list, get_val_train_ds
from utils.const import DS_ROOT_PTH, LOG_DIR

# define super-params

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

dict_plain_ds_train, dict_plain_ds_test = get_val_train_ds(DS_ROOT_PTH)

# =================================================
# get train dataset paths and labels
# =================================================
all_labels = dict_plain_ds_train['labels']
n_dataset_len = len(all_labels)
x = reduce(lambda _x, _y: _x + ([_y] if _y not in _x else []), all_labels, [])
n_classes = len(x)
print("Have n_classes: %d !" % n_classes)

print(dict_plain_ds_train['labels'])
dict_plain_ds_train['labels'] = map(lambda _T: x.index(_T), dict_plain_ds_train['labels'])
print(dict_plain_ds_train['labels'])
dict_plain_ds_train['labels'] = tf.keras.utils.to_categorical(dict_plain_ds_train['labels'], num_classes=n_classes)
print(dict_plain_ds_train['labels'])

# =================================================
# get test dataset paths and labels
# =================================================
all_labels_test = dict_plain_ds_test['labels']
n_test_len = len(all_labels_test)
x_test = reduce(lambda _x, _y: _x + ([_y] if _y not in _x else []), all_labels_test, [])
n_classes_test = len(x_test)
print("Have n_classes_test: %d !" % n_classes_test)

print(dict_plain_ds_test['labels'])
dict_plain_ds_test['labels'] = map(lambda _T: x_test.index(_T), dict_plain_ds_test['labels'])
print(dict_plain_ds_test['labels'])
dict_plain_ds_test['labels'] = tf.keras.utils.to_categorical(dict_plain_ds_test['labels'], num_classes=n_classes_test)
print(dict_plain_ds_test['labels'])
# exit(-1)

# define module of entrance
yggdrasil = Yggdrasil(n_class=n_classes, n_dataset_len=n_dataset_len)


def _parser_ds(dict_ds_item):
    # print(dict_ds_item)
    t_image = dict_ds_item['images']
    t_label = dict_ds_item['labels']

    t_img_str = tf.read_file(t_image)
    t_img_decoded = tf.image.decode_jpeg(t_img_str)
    t_img_resized = tf.image.resize_image_with_crop_or_pad\
        (t_img_decoded, target_height=Yggdrasil.in_height, target_width=Yggdrasil.in_width)

    # t_label = tf.cast(t_label, dtype=tf.int16)
    # t_label = tf.keras.utils.to_categorical(t_label, num_classes=Yggdrasil.n_class)

    print(t_img_resized)
    print(t_label)
    return {'image': t_img_resized, 'label': t_label}  # 注意这里我更改了没一个维度的变量的key的内容


# define datasets
ds_train = tf.data.Dataset.from_tensor_slices(dict_plain_ds_train)
ds_train = ds_train.map(_parser_ds)
ds_train = ds_train.shuffle(buffer_size=1024, reshuffle_each_iteration=True)\
    .batch(Yggdrasil.batch_size)\
    .repeat(Yggdrasil.epoch)

ds_test = tf.data.Dataset.from_tensor_slices(dict_plain_ds_test)
ds_test = ds_test.map(_parser_ds)
ds_test = ds_test.batch(n_test_len)


iterator = ds_train.make_one_shot_iterator()
iterator_test = ds_test.make_one_shot_iterator()

# with tf.Session() as sess:
#     m = sess.run(iterator_test.get_next())
#     print(m)
#     print(m['image'].shape)
#
#
# exit(-1)


with tf.Session() as sess:

    global_step = tf.Variable(0, trainable=False)

    X = tf.placeholder(tf.float32, shape=(None, Yggdrasil.in_height, Yggdrasil.in_width, Yggdrasil.in_channel))
    Y = tf.placeholder(tf.float32, shape=(None, yggdrasil.n_class))

    with tf.name_scope('session'):
        with tf.name_scope('logits'):
            logits = yggdrasil.model(X)
            tf.summary.histogram("logits", logits)
            # tf.summary.scalar("logits", logits)

        with tf.name_scope('inferences'):
            inference = tf.nn.softmax(logits=logits)

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))
            tf.summary.histogram("loss", loss)
            # tf.summary.scalar("loss", loss)

        with tf.name_scope('optimizer'):
            # decay=5e-4,
            optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, use_nesterov=True).\
                minimize(loss, global_step=global_step,)

        with tf.name_scope('summary_ops'):
            summaries = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
            sum_writer = tf.summary.FileWriter(logdir=LOG_DIR, graph=sess.graph)

        with tf.name_scope('get_next_element_in_ds'):
            next_element = iterator.get_next()
            next_element_in_valid = iterator_test.get_next()

    # initialize
    init = tf.global_variables_initializer()
    sess.run(init)

    with tf.name_scope('saver_ops'):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(LOG_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore model')
            saver.restore(sess, ckpt.model_checkpoint_path)

    print("======== Training Begin ========")
    while global_step.eval() * Yggdrasil.batch_size < Yggdrasil.epoch * Yggdrasil.n_dataset_len:
        run_metadata = tf.RunMetadata()

        tmp_recode = sess.run(next_element)
        _, cal_loss, sums = sess.run([optimizer, loss, summaries], feed_dict={X: tmp_recode['image'], Y: tmp_recode['label']})

        sum_writer.add_summary(sums, global_step.eval())
        if global_step.eval() % 100 == 0:
            predicts = sess.run([inference,], feed_dict={X: tmp_recode['image']})
            acc = cal_acc(predicts, tmp_recode['label'])

            saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt.'+str(global_step.eval())))
            print("%s : epoch:[%d] - step:[%d] | with loss [%.8f] and acc:[%.5f] percentage" %
                  (curr_normal_time(), (global_step.eval()*Yggdrasil.batch_size/Yggdrasil.n_dataset_len),
                   global_step.eval(), cal_loss, acc*100))
            sum_writer.add_run_metadata(run_metadata, tag="step:%d"%global_step.eval(), global_step=global_step.eval())

    print("======== Training Finished ========")

    print("======== Testing Begin ========")
    tmp_test_record = sess.run(iterator_test.get_next())
    test_X = tmp_test_record['image']
    test_Y = tmp_test_record['label']
    print("load test data input X with shape: "+str(test_X.shape))
    prediction, cal_loss = sess.run([inference, loss], feed_dict={X: test_X, Y: test_Y})
    print prediction
    print test_Y
    acc = cal_acc(prediction, test_Y)
    print("Acc: %.7f" % acc)
    print("Loss: %.8f" % cal_loss)






