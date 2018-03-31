#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 29/03/2018

import os


def get_data_list(root_pth):
    _dataset_train = {'images': [], 'labels': []}
    _dataset_test = {'images': [], 'labels': []}

    train_pth = os.path.join(root_pth, "bounding_box_train")
    # val_pth = os.path.join(root_pth, "val")
    test_pth = os.path.join(root_pth, "bounding_box_test")

    for root, dirs, files in os.walk(train_pth, topdown=True):
        # print(root)
        # print(dirs)
        # print(files)
        for file_name in files:
            if not file_name[-4:] == '.jpg':
                continue
            fsplits = file_name.split('_')
            tmp_label = fsplits[0]
            _dataset_train['images'].append(os.path.join(root, file_name))
            _dataset_train['labels'].append(int(tmp_label))  # 强行增加了一个label的数字化的代码

    for root, dirs, files in os.walk(test_pth, topdown=True):
        for file_name in files:
            if not file_name[-4:] == '.jpg':
                continue
            fsplits = file_name.split('_')
            tmp_label = fsplits[0]
            if tmp_label == '-1':
                # 暂时去掉-1的 保证751类
                continue
            _dataset_test['images'].append(os.path.join(root, file_name))
            _dataset_test['labels'].append(int(tmp_label))  # 强行增加了一个label的数字化的代码

    return _dataset_train, _dataset_test

