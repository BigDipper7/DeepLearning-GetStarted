#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 12/10/2017

from scipy.misc import imread, imresize, imsave
import numpy as np

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


print "======"

target = imread(img_path, mode="RGB")

target = target.transpose((2, 0, 1))

print target
print target.shape

# target = np.expand_dims(target, axis=0)

print target.shape

target_R = target[0]
target_G = target[1]
target_B = target[2]

final_R = np.array([target_R, np.zeros(target_G.shape), np.zeros(target_B.shape)])
print final_R
print final_R.shape
final_R = final_R.transpose((1, 2, 0))
imsave(base_path+"/003_final_R.png", final_R)


final_G = np.array([np.zeros(target_R.shape), target_G, np.zeros(target_B.shape)])
print final_G
print final_G.shape
final_G = final_G.transpose((1, 2, 0))
imsave(base_path+"/003_final_G.png", final_G)


final_B = np.array([np.zeros(target_R.shape), np.zeros(target_G.shape), target_B])
print final_B
print final_B.shape
final_B = final_B.transpose((1, 2, 0))
imsave(base_path+"/003_final_B.png", final_B)
