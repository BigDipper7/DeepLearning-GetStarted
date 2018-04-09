#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: violinsolo
# Created on 28/12/2017

import numpy as np
import os
import h5py
import time


# =====================
# =     Data Module   =
# =====================

# parse string to boolean
def str_to_boolean(v):
    return v.lower() in ("true", "yes", "t", "1")


# calculate acc with predicts_labels and one-hot_softmax_labels
# of a shape *[None, ONE_HOT_LABELS_CLASS_NUM]*
def cal_acc(t_predict, t_labels):
    predict_label = np.argmax(t_predict, axis=-1)
    ground_truth_label = np.argmax(t_labels, axis=-1)

    correct_num = 0
    total_num = len(ground_truth_label)

    for i in range(total_num):
        if predict_label[i] == ground_truth_label[i]:
            correct_num += 1
            print(i)

    acc = float(correct_num) / total_num
    # print("Acc: %.7f" % acc)
    return acc


# =====================
# =     Time Module   =
# =====================
# -----------
#     %y 两位数的年份表示（00-99）
#     %Y 四位数的年份表示（000-9999）
#     %m 月份（01-12）
#     %d 月内中的一天（0-31）
#     %H 24小时制小时数（0-23）
#     %I 12小时制小时数（01-12）
#     %M 分钟数（00=59）
#     %S 秒（00-59）
#     %a 本地简化星期名称
#     %A 本地完整星期名称
#     %b 本地简化的月份名称
#     %B 本地完整的月份名称
#     %c 本地相应的日期表示和时间表示
#     %j 年内的一天（001-366）
#     %p 本地A.M.或P.M.的等价符
#     %U 一年中的星期数（00-53）星期天为星期的开始
#     %w 星期（0-6），星期天为星期的开始
#     %W 一年中的星期数（00-53）星期一为星期的开始
#     %x 本地相应的日期表示
#     %X 本地相应的时间表示
#     %Z 当前时区的名称
#     %% %号本身
# -----------

# -----------
# 常用方法
# -----------


def time_with_regex(regex, target):
    """
    正则化时间格式并打印出来
    :param regex: 正则内容详见上方注释
    :param target: time object
    :return: str of regex time
    """
    return time.strftime(regex, target)


def de_timestr_with_regex_to_timestamp(timestr, regex):
    """
    将格式字符串转换为时间戳
    :param timestr: 某种格式的输入字符串
    :param regex: 正则串
    :return: 时间戳
    """
    return time.mktime(time.strptime(timestr, regex))


def curr_timestamp_time():
    return time.time()


def curr_normal_time():
    """
    格式化成2016-03-20 11:45:39形式
    :return: regex: {%Y-%m-%d %H:%M:%S}
    """
    return curr__yyyy_mm_dd_HH_MM_SS__time()


def time_span(before):
    """
    获得和before相比的时间差
    :param before: 前一个时间
    :return: float32, 时间差，单位 秒（s）
    """
    after = curr_timestamp_time()
    return after - before

# -----------
# 非常用方法
# -----------


def _curr_time_with_regex(regex):
    """
    获得当前时间的正则格式并打印出来
    :param regex: 正则内容详见上方注释
    :return: str of regex current time
    """
    return time_with_regex(regex, time.localtime())


def curr__yyyy_mm_dd_HH_MM_SS__time():
    """
    格式化成2016-03-20 11:45:39形式
    :return: regex: {%Y-%m-%d %H:%M:%S}
    """
    return _curr_time_with_regex("%Y-%m-%d %H:%M:%S")


def curr__a_b_d_HH_MM_SS_yyyy__time():
    """
    格式化成Sat Mar 28 22:24:24 2016形式
    :return: regex: {%a %b %d %H:%M:%S %Y}
    """
    return _curr_time_with_regex("%a %b %d %H:%M:%S %Y")


def de_time_to_obj(timestr):
    """
    将格式字符串转换为时间戳
    'Sat Mar 28 22:24:24 2016' 转化为时间戳
    :param timestr: 'Sat Mar 28 22:24:24 2016'格式的输入数据
    :return: 时间戳
    """
    return de_timestr_with_regex_to_timestamp(timestr, "%a %b %d %H:%M:%S %Y")


# =====================
# =     END Module    =
# =====================
