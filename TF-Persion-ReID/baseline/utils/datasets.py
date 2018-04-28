#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 29/03/2018

import os
import tensorflow as tf
import numpy as np
from const import IN_HEIGHT, IN_WIDTH, SEED, BATCH_SIZE, EPOCH, DS_ROOT_PTH, DS_ROOT_PTH_PYTORCH


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

    for root, dirs, files in os.walk(_valid_path, topdown=True):
        print(root)
        print(dirs)
        print(files)

        for str_class_name in dirs:
            # 遍历所有的class name，即包含的dir们
            _tmp_valid_set_pth = os.path.join(root, str_class_name)

            for c_root, c_dirs, c_files in os.walk(_tmp_valid_set_pth, topdown=True):
                for c_file in c_files:
                    if not c_file[-4:] == '.jpg':
                        continue
                    # 遍历获取每次的file name，用于拼接出image的path
                    _dataset_valid['images'].append(os.path.join(c_root, c_file))
                    _dataset_valid['labels'].append(int(str_class_name))  # 强行增加了一个label的数字化的代码
                # 结束当前循环，后面没意义了
                break

        # 直接结束循环，后面的循环已经没有什么意义了
        break

    return _dataset_train, _dataset_valid


def _get_plain_ds_info():
    """
    获取原始数据集plain信息，以dict的形式进行存储
    :return: dicts '{"images": [], "pids": [], "cids": []}'
    """
    plain_ds_train = {"images": [], "pids": [], "cids": []}
    plain_ds_valid = {"images": [], "pids": [], "cids": []}

    _train_path = os.path.join(DS_ROOT_PTH, "pytorch/train")
    _valid_path = os.path.join(DS_ROOT_PTH, "pytorch/val")

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
                    plain_ds_train['images'].append(os.path.join(c_root, c_file))
                    plain_ds_train['pids'].append(int(str_class_name))  # 强行增加了一个label的数字化的代码
                    plain_ds_train['cids'].append(int(str(c_file).split('_')[1][1]))  # 强行增加了一个cid的数字化的代码
                # 结束当前循环，后面没意义了
                break

        # 直接结束循环，后面的循环已经没有什么意义了
        break

    for root, dirs, files in os.walk(_valid_path, topdown=True):
        print(root)
        print(dirs)
        print(files)

        for str_class_name in dirs:
            # 遍历所有的class name，即包含的dir们
            _tmp_valid_set_pth = os.path.join(root, str_class_name)

            for c_root, c_dirs, c_files in os.walk(_tmp_valid_set_pth, topdown=True):
                for c_file in c_files:
                    if not c_file[-4:] == '.jpg':
                        continue
                    # 遍历获取每次的file name，用于拼接出image的path
                    plain_ds_valid['images'].append(os.path.join(c_root, c_file))
                    plain_ds_valid['pids'].append(int(str_class_name))  # 强行增加了一个label的数字化的代码
                    plain_ds_valid['cids'].append(int(str(c_file).split('_')[1][1]))  # 强行增加了一个cid的数字化的代码
                # 结束当前循环，后面没意义了
                break

        # 直接结束循环，后面的循环已经没有什么意义了
        break

    return plain_ds_train, plain_ds_valid


def _re_group_plain_ds_info():
    """
    对原始数据再次整合成新的dict
    :return: dicts "{'images': [], 'labels': [], 'pids': [], 'cids': []}"
        其中images存储了image的file path的list，         @string
        labels存储了one-hot的label的represent的list，    @binary list with shape:(n_classes,)
        pids存储了原始的lable的信息，用于鉴别是否是同一个人， @int32
        cids存储了camera的信息，用于鉴别识别效果，          @int32
    """
    plain_ds_train, plain_ds_valid = _get_plain_ds_info()

    # -----
    # process train ds info
    all_pids = plain_ds_train['pids']
    no_dupl_pids = sorted(set(all_pids))

    n_classes_train = len(no_dupl_pids)

    one_hot_pids = tf.keras.utils.to_categorical(all_pids, num_classes=n_classes_train)
    plain_ds_train['labels'] = one_hot_pids

    # -----
    # process valid ds info
    all_pids = plain_ds_valid['pids']
    no_dupl_pids = sorted(set(all_pids))

    n_classes_valid = len(no_dupl_pids)

    one_hot_pids = tf.keras.utils.to_categorical(all_pids, num_classes=n_classes_valid)
    plain_ds_valid['labels'] = one_hot_pids

    return n_classes_train, plain_ds_train, plain_ds_valid


def _ds_mapping_parser(item):
    plain_images = item['images']
    plain_labels = item['labels']
    plain_pids = item['pids']
    plain_cids = item['cids']

    # ----
    # process images
    img_bytes = tf.read_file(plain_images)
    img_decoded = tf.image.decode_jpeg(img_bytes, channels=3)
    img_resized = tf.image.resize_images(images=img_decoded, size=(IN_HEIGHT, IN_WIDTH), method=tf.image.ResizeMethod.BILINEAR)
    img_rand_flip = tf.image.random_flip_left_right(img_resized, seed=SEED)

    img_norm = tf.image.per_image_standardization(img_rand_flip)

    return {'image': img_norm, 'label': plain_labels, 'pid': plain_pids, 'cid': plain_cids}


def get_ds_iterators():
    '''
    获得dataset iterators
    :return: n_classes_train, train_os_iterator, valid_os_iterator
    '''
    n_classes_train, plain_ds_train, plain_ds_valid = _re_group_plain_ds_info()

    ds_train = tf.data.Dataset.from_tensor_slices(plain_ds_train)
    ds_train = ds_train.map(_ds_mapping_parser)
    ds_train = ds_train.batch(BATCH_SIZE).repeat(EPOCH).shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
    train_os_iterator = ds_train.make_one_shot_iterator()

    ds_valid = tf.data.Dataset.from_tensor_slices(plain_ds_valid)
    ds_valid = ds_valid.map(_ds_mapping_parser)
    ds_valid = ds_valid.batch(BATCH_SIZE).repeat(EPOCH).shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
    valid_os_iterator = ds_valid.make_one_shot_iterator()

    return n_classes_train, train_os_iterator, valid_os_iterator


if __name__ == '__main__':
    from const import DS_ROOT_PTH
    # x_ds_train, x_ds_valid = get_val_train_ds(DS_ROOT_PTH)

    x_ds_train, x_ds_valid = _get_plain_ds_info()
    print(x_ds_train, x_ds_valid)