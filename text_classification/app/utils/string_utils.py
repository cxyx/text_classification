#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yang Huiyu
# @Date  : 2019/5/16
from __future__ import unicode_literals

import re

PUNCTUATIONS = [
    ',', '，', '。', '.', '；', ';', '：', ':', '"', '{', '}', '[', ']', '【', '】',
    '(', ')', '@', '%', '％', '*', '↗', '※', '．', '<', '>', '《', '》', '!', '！',
    '?', '？', '-', '―', '—', '_', ' ', '/', '\\', '、', '\n', '\t', '“', '”', '"', '#',
    '（', '）', '+', '~', '…', '　', '|', '‘', '’', '=', '\'', '&'
]


def check_contain_english(uchars):
    for uchar in uchars:
        """判断一个unicode是否英文"""
        if (u'\u005a' >= uchar >= u'\u0041') or (u'\u007a' >= uchar >= u'\u0061'):
            return True
    return False


def check_contain_number(uchars):
    for uchar in uchars:
        """判断一个unicode是否是数字"""
        if u'\u0039' >= uchar >= u'\u0030':
            return True
    return False


def check_contain_punctuation(uchars):
    for uchar in uchars:
        """判断一个unicode是否为标点"""
        if uchar in PUNCTUATIONS:
            return True
    return False


def filter_brackets(content):
    """
    去除括号里的内容，只处理一层，不管嵌套的括号。

    :param content:
    :return:
    """
    content = re.sub(r'\([^()]*\)', '', content)
    content = re.sub(r'\[[^[]*\]', '', content)
    content = re.sub(r'（[^（）]*）', '', content)
    content = re.sub(r'{[^{}]*}', '', content)
    return content


def count_pattern(content, pattern):
    """
    count sub-str fits given pattern

    :param content: str, content to search pattern
    :param pattern: regex, pattern to match
    :return: int, num of sub-str fits pattern
    """
    count = len(re.findall(pattern, content))
    return count
