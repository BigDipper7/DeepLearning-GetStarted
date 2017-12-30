#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 12/12/2017

import tensorflow as tf


# define FIFO queue
q = tf.FIFOQueue(capacity=1000, dtypes='float32')

# define ops
counter = tf.Variable(initial_value=0, dtype='float32')
counter_increment_op = tf.assign_add(counter, 1.)
queue_enqueue_op = q.enqueue([counter])

coordinator = tf.train.Coordinator()

qr = tf.train.QueueRunner(queue=q, enqueue_ops=[counter_increment_op, queue_enqueue_op])

# begin session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    enqueue_threads = qr.create_threads(sess=sess, start=True, coord=coordinator)


    for i in range(10):
        print sess.run(q.dequeue())

    # coordinator.join(enqueue_threads)

    coordinator.request_stop()
    print sess.run(q.size())
    coordinator.join(enqueue_threads)
    for i in range(100):
        print "-%d-" % i
    print sess.run(q.size())
