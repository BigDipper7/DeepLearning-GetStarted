#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 15/12/2017

import tensorflow as tf
from utils import util

# prepare data
from tensorflow.examples.tutorials.mnist import input_data

print 'Preparing data...'
# 下载一直失败，查看源代码原来是access denied，利用镜像搞定
# CVDF mirror of http://yann.lecun.com/exdb/mnist/
# SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
train = mnist.train
valid = mnist.validation
test = mnist.test
print 'Data prepared...'

# define hyper-params for network
learning_rate = 0.001
iter_len = 200000
batch_size = 128
dropout = 0.75

# define network
n_input_size = 28 * 28
n_output_classes_size = 10

# define global params
MODEL_PATH = './models'
MODEL_SAVE_SPAN = 100


# define tf graph func
def conv2d(X, fil, stride, b, padding='SAME', name='conv_default'):
    conv = tf.nn.conv2d(input=X, filter=fil, strides=[1, stride, stride, 1], padding=padding, name=name)
    x_b = tf.nn.bias_add(conv, bias=b)
    y = tf.nn.relu(x_b)
    return y


def maxpooling2d(X, k, stride, padding='SAME', name='maxpool_default'):
    maxpool = tf.nn.max_pool(X, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding=padding, name=name)
    return maxpool


def norm4d(X, radius, name='norm_default'):
    norm = tf.nn.lrn(X, depth_radius=radius, bias=1.0, alpha=0.001/9.0, beta=0.75, name=name)
    return norm


# define AlexNet Params
weights = {                                                   # [None, 28, 28, 1]
    'c1': tf.Variable(tf.random_normal([11, 11, 1, 96])),     # [None, 28, 28, 96]
    'm1': 2,                                                  # [None, 14, 14, 96]
    'n1': 4,                                                  # [None, 14, 14, 96]
    'c2': tf.Variable(tf.random_normal([5, 5, 96, 256])),     # [None, 14, 14, 256]
    'm2': 2,                                                  # [None, 7, 7, 256]
    'n2': 4,                                                  # [None, 7, 7, 256]
    'c3': tf.Variable(tf.random_normal([3, 3, 256, 384])),    # [None, 7, 7, 384]
    'm3': 2,                                                  # [None, 4, 4, 384]
    'n3': 4,                                                  # [None, 4, 4, 384]
    'c4': tf.Variable(tf.random_normal([3, 3, 384, 384])),    # [None, 4, 4, 384]
    'c5': tf.Variable(tf.random_normal([3, 3, 384, 256])),    # [None, 4, 4, 256]
    'm5': 2,                                                  # [None, 4, 4, 256]
    'n5': 4,                                                  # [None, 4, 4, 256]
    'f1': tf.Variable(tf.random_normal([4*4*256, 4096])),     # [None, 4096]
    'f2': tf.Variable(tf.random_normal([4096, 4096])),        # [None, 4096]
    'out': tf.Variable(tf.random_normal([4096, 10])),         # [None, 10]                                                     # [None, 4, 4, 256]
}

strides = {
    'c1': 1,
    'm1': 2,
    'c2': 1,
    'm2': 2,
    'c3': 1,
    'm3': 2,
    'c4': 1,
    'c5': 1,
    'm5': 2,
}

bias = {                                          # [None, 28, 28, 1]
    'c1': tf.Variable(tf.random_normal([96])),    # [None, 28, 28, 96]
    'c2': tf.Variable(tf.random_normal([256])),   # [None, 28, 28, 256]
    'c3': tf.Variable(tf.random_normal([384])),   # [None, 14, 14, 384]
    'c4': tf.Variable(tf.random_normal([384])),   # [None, 7, 7, 384]
    'c5': tf.Variable(tf.random_normal([256])),   # [None, 7, 7, 256]
    'f1': tf.Variable(tf.random_normal([4096])),  # [None, 4096]
    'f2': tf.Variable(tf.random_normal([4096])),  # [None, 4096]
    'out': tf.Variable(tf.random_normal([10])),   # [None, 10]
}


# define network
def alexnet(X, weight, b, stride, keep_prob):
    X = tf.reshape(X, [-1, 28, 28, 1], name='inputX')
    # [None, 28, 28, 1]

    conv1 = conv2d(X, weight['c1'], stride['c1'], b['c1'], name='conv1')
    # [None, 28, 28, 96]
    maxpool1 = maxpooling2d(conv1, weight['m1'], stride['m1'], name='maxpool1')
    # [None, 14, 14, 96]
    norm1 = norm4d(maxpool1, 5, name='norm1')
    # [None, 14, 14, 96]

    conv2 = conv2d(norm1, weight['c2'], stride['c2'], b['c2'], name='conv2')
    # [None, 14, 14, 256]
    maxpool2 = maxpooling2d(conv2, weight['m2'], stride['m2'], name='maxpool2')
    # [None, 7, 7, 256]
    norm2 = norm4d(maxpool2, 5, name='norm2')
    # [None, 7, 7, 256]

    conv3 = conv2d(norm2, weight['c3'], stride['c3'], b['c3'], name='conv3')
    # [None, 7, 7, 384]
    maxpool3 = maxpooling2d(conv3, weight['m3'], stride=stride['m3'], name='maxpool3')
    # [None, 4, 4, 384]
    norm3 = norm4d(maxpool3, 5, name='norm3')
    # [None, 4, 4, 384]

    conv4 = conv2d(norm3, weight['c4'], stride['c4'], b['c4'], name='conv4')
    # [None, 4, 4, 384]
    conv5 = conv2d(conv4, weight['c5'], stride['c5'], b['c5'], name='conv5')
    # [None, 4, 4, 384]
    #maxpool5 = maxpooling2d(conv5, weight['m5'], stride['m5'])
    norm5 = norm4d(conv5, 5, name='norm5')
    # [None, 2, 2, 256]

    fc1 = tf.reshape(norm5, [-1, weight['f1'].get_shape().as_list()[0]], name='reshape_to_vector')
    # [None, 4 * 4 * 256]
    fc1 = tf.matmul(fc1, weight['f1'])
    # [None, 4096]
    fc1 = tf.nn.bias_add(fc1, b['f1'])
    fc1 = tf.nn.relu(fc1, name='relu_fc1')
    # [None, 4096]

    # dropout
    drop1 = tf.nn.dropout(fc1, keep_prob=keep_prob)
    # [None, 4096]

    fc2 = tf.nn.bias_add(tf.matmul(tf.reshape(drop1, [-1, weight['f2'].get_shape().as_list()[0]]), weight['f2']), b['f2'])
    fc2 = tf.nn.relu(fc2)
    # [None, 4096]

    # dropout
    drop2 = tf.nn.dropout(fc2, keep_prob=keep_prob)
    # [None, 4096]

    out = tf.nn.bias_add(tf.matmul(drop2, weight['out']), b['out'])
    # [None, 10]

    return out


with tf.Session() as sess:
    import os
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
        print 'mkdir {%s}' % MODEL_PATH
    global_step = tf.Variable(0, trainable=False)
    print 'prepared global_sept setting finished....'

    # init = tf.initialize_all_variables()  # 不用在上面一句话的原因是会造成FailedPreconditionError，原因是没有初始化完成
    x = tf.placeholder(dtype='float32', shape=[None, n_input_size], name='oriX')
    y = tf.placeholder(dtype='float32', shape=[None, n_output_classes_size], name='oriY')
    keep_prob = tf.placeholder(dtype='float32', name='oriKeepProb')
    batch_x = []  # to access out of the loop
    batch_y = []

    # define network
    logits = alexnet(x, weights, bias, strides, keep_prob)
    # calculate loss
    # loss = tf.reduce_mean(y - pred) #TODO: you wen ti
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)  # 处理好多异常情况，比如说括号里的是ndarray
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    # predict
    # show whether MAX arg index is matched
    corr_pred = tf.equal(tf.arg_max(logits, 1), tf.arg_max(y, 1))  # arg_max return index!
    # get a Tensor of [None, 1], and '1' is a array of [True, False]. Result is like [True, False, False, True, True]
    accuracy = tf.reduce_mean(tf.cast(corr_pred, dtype='float32'))  # Firstly, cast True/False to 1./0.; Then, calculate the mean
    # get a num of dtype=np.float32 represents the accurate

    saver = tf.train.Saver(max_to_keep=10)
    print 'All Definitions are prepared....'

    print 'Begin to prepare initializer'
    init = tf.global_variables_initializer()  # IMPORTENT: init要写在这里，原因是AdamOptimizer的minimize的时候会产生额外的tensor，但是你的init写的太前了，所以不管用
    sess.run(init)
    print 'Initializer preparation finished....'

    print ' ============================================== '
    print ' =               begin training               = '
    print ' ============================================== '
    while global_step.eval() * batch_size < iter_len:
        before = util.curr_timestamp_time()

        batch_x, batch_y = train.next_batch(batch_size=batch_size)
        if global_step.eval() == 0:
            print "SHAPE: batch_x %s, batch_y %s" % (batch_x.shape, batch_y.shape)

        # training...
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        print 'Train Step: %d optimized, cost %.3fS. ' % (global_step.eval(), util.time_span(before))

        if global_step.eval() % MODEL_SAVE_SPAN == 0:
            saver.save(sess, MODEL_PATH+'/model.ckpt', global_step)  # need to change it to the prefix of model
            print 'Save model at: %s' % util.curr_normal_time()

            start_pred_timestamp = util.curr_timestamp_time()
            los, acc = sess.run([loss, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})  # cal the loss and acc
            time_span = util.time_span(start_pred_timestamp)
            avg_los = sess.run(tf.reduce_mean(los))
            print 'Model Ability: current - total loss: %s,\n -- avg_loss: %s, acc: %.8f' % (los, avg_los, acc)

        # print 'Step: %d finished, cost %.3fS. ' % (global_step.eval(), util.time_span(before))
        global_step.assign_add(1)

    print ' ============================================== '
    print ' =               Finished Training            = '
    print ' ============================================== '

    print ' ============================================== '
    print ' =               begin validating             = '
    print ' ============================================== '

    start_pred_timestamp = util.curr_timestamp_time()
    los, acc = sess.run([loss, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
    span = util.time_span(start_pred_timestamp)

    avg_los = sess.run(tf.reduce_mean(los))
    print 'Model Ability: current - total loss: %s,\n -- avg_loss: %s, acc: %.8f, time: %.5fS' % (los, avg_los, acc, span)

    print ' ============================================== '
    print ' =               begin testing                = '
    print ' ============================================== '
    start_pred_timestamp = util.curr_timestamp_time()
    los, acc = sess.run([loss, accuracy], feed_dict={x: test.images[:512], y: test.labels[:512], keep_prob: 1.})
    span = util.time_span(start_pred_timestamp)

    avg_los = tf.reduce_mean(los)
    print 'Testing: Model Performance: current - total loss: %s,\n -- avg_loss: %s, acc: %.8f, time: %.5fS' % (los, avg_los, acc, span)

