#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 12/10/2017

from scipy.misc import imread, imresize
import numpy as np
import os
import h5py


# parse string to boolean
def str_to_boolean(v):
    return v.lower() in ("true", "yes", "t", "1")


# util function to open, resize and format pictures into appropriate tensors
def pre_process_image(image_path, img_width=100, img_height=100, load_dims=False):
    global img_WIDTH, img_HEIGHT, aspect_ratio

    img = imread(image_path, mode="RGB")  # Prevents crashes due to PNG images (ARGB)
    if load_dims:
        img_WIDTH = img.shape[0]
        img_HEIGHT = img.shape[1]
        aspect_ratio = img_HEIGHT / img_WIDTH

    img = imresize(img, (img_width, img_height))
    img = img.transpose((2, 0, 1)).astype('float64')
    img = np.expand_dims(img, axis=0)
    return img


# util function to convert a tensor into a valid image
def de_process_image(x):
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def load_weights(weight_path, model):
    assert os.path.exists(weight_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weight_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')


# # the gram matrix of an image tensor (feature-wise outer product)
# def gram_matrix(x):
#     assert K.ndim(x) == 3
#     features = K.batch_flatten(x)
#     gram = K.dot(features, K.transpose(features))
#     return gram
#
# def eval_loss_and_grads(x):
#     x = x.reshape((1, 3, img_width, img_height))
#     outs = f_outputs([x])
#     loss_value = outs[0]
#     if len(outs[1:]) == 1:
#         grad_values = outs[1].flatten().astype('float64')
#     else:
#         grad_values = np.array(outs[1:]).flatten().astype('float64')
#     return loss_value, grad_values