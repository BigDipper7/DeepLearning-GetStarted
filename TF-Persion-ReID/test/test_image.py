#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 12/10/2017

from scipy.misc import imread, imresize, imsave

base_path = "/Users/violinsolo/dev/dataset/image"
img_path = base_path + "/003.png"

target = imread(img_path)

print target

print "======"

print target.shape

print "======"

print target[0]
print target[0].shape

print "======"

print target[0][0]
print target[0][0].shape


print "---------------"
target = imread(img_path, flatten=True, mode="RGB")

print target

print "======"

print target.shape

print "======"

print target[0]
print target[0].shape

print "======"

print target[0][0]
print target[0][0].shape

target = imresize(target, size=(1080, 1920, 3), interp="bicubic")

imsave(base_path + "/target.png", target)
