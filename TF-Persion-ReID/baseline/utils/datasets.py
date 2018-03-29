#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 29/03/2018

import os


def get_data_list(root_pth):
    _dataset_train = {'images': [], 'labels': []}

    train_pth = os.path.join(root_pth, "bounding_box_train")
    # val_pth = os.path.join(root_pth, "val")
    test_pth = os.path.join(root_pth, "bounding_box_test")

    for root, dirs, files in os.walk(train_pth, topdown=True):
        print(root)
        print(dirs)
        print(files)
        for file_name in files:
            if not file_name[-4:] == '.jpg':
                continue
            fsplits = file_name.split('_')
            tmp_label = fsplits[0]
            _dataset_train['images'].append(os.path.join(root, file_name))
            _dataset_train['labels'].append(os.path.join(tmp_label))

    return _dataset_train

