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


def get_val_train_ds(split_root_pth):
    _dataset_train = {"images": [], "labels": []}
    _dataset_valid = {"images": [], "labels": []}

    _train_path = os.path.join(split_root_pth, "pytorch/train")
    _valid_path = os.path.join(split_root_pth, "pytorch/val")

    for root, dirs, files in os.walk(_train_path, topdown=True):
        print(root)
        print(dirs)
        print(files)

        for str_class_name in dirs:
            # 遍历所有的class name，即包含的dir们
            _tmp_train_set_pth = os.path.join(root, str_class_name)

            for c_root, c_dirs, c_files in os.walk(_tmp_train_set_pth, topdown=True):
                for c_file in c_files:
                    if not c_file[-4:] == '.jpg':
                        continue
                    # 遍历获取每次的file name，用于拼接出image的path
                    _dataset_train['images'].append(os.path.join(c_root, c_file))
                    _dataset_train['labels'].append(int(str_class_name))  # 强行增加了一个label的数字化的代码
                # 结束当前循环，后面没意义了
                break

        # 直接结束循环，后面的循环已经没有什么意义了
        break


    return _dataset_train



if __name__ == '__main__':
    from const import DS_ROOT_PTH
    x_ds_train = get_val_train_ds(DS_ROOT_PTH)