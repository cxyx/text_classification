#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yang Huiyu
# @Date  : 2019/5/16

from __future__ import unicode_literals

import numpy as np
from cachetools import cached, LRUCache

from string_utils import check_contain_number, check_contain_english, check_contain_punctuation

all_feature_types = ['ngram']


def ngram(content, gram_num=2, filter_num=False, filter_eng=False, filter_punc=False):
    result = [
        content[i:i + gram_num] for i in xrange(len(content) - gram_num + 1)
        if not (filter_num and check_contain_number(content[i:i + gram_num]))
        if not (filter_eng and check_contain_english(content[i:i + gram_num]))
        if not (filter_punc and check_contain_punctuation(content[i:i + gram_num]))
    ]
    return result


@cached(cache=LRUCache(maxsize=10000))
def get_ner_type(term, ner_obj, types=None):
    """
    判断term属于人名、地名、机构名

    :param term:
    :type term: unicode
    :param ner_obj:
    :type ner_obj:
    :param types:
    :type types: list[str]
    :return:
    """
    if types is None:
        types = ['person']
    if len(term) < 2 or len(term) > 3 or 'n' in term:
        return ''
    for value, _, _, ner_type in ner_obj.find_by_types(term, types=types, model='crf'):
        return ner_type
    return ''


def get_points_interval(points, interval_len, min_ratio=0.8):
    """
    获取点的集中分布区间

    :param points: 点的分布
    :type points: list
    :param interval_len: 区间长度
    :type interval_len: float
    :param min_ratio: 区间内点的最小占比
    :type min_ratio: float
    :return: 
    """
    if not points:
        raise ValueError('参数points不能为空!'.encode('utf-8'))
    if interval_len > 1 or interval_len <= 0:
        raise ValueError('参数interval_len必须在(0, 1]之间!'.encode('utf-8'))
    if min_ratio > 1 or min_ratio < 0:
        raise ValueError('参数min_ration必须在[0, 1]之间!'.encode('utf-8'))

    median = np.median(points)
    start = max([0, median - interval_len / 2.])
    end = min([1, median + interval_len / 2.])

    ratio = len([p for p in points if start <= p <= end]) / float(len(points))

    return (start, end) if ratio >= min_ratio else None


def gen_one_hot_vec(words, word_index_dict, vec_len):
    """generate one-hot encoding vector

    :param words: content to be encode by one-hot
    :type words: list[unicode]
    :param word_index_dict: dict of feature words, <unknown> at the end
    :type word_index_dict: dict[unicode, int]
    :param vec_len: num of words in word_index_dict
    :type vec_len: int
    :return: one-hot vector
    :rtype: list[int]
    """
    feature_vec = [0] * vec_len
    for word in words:
        index = word_index_dict.get(word, vec_len - 1)
        feature_vec[index] = 1
    return feature_vec
