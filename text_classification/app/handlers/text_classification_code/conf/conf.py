#! /usr/bin/env python
# coding:utf-8

import os

CUR_PATH =  os.path.join(os.getcwd(), '../conf/')#os.path.dirname(__file__)

TRAIN_FILE_PATH = os.path.abspath(os.path.join(
    CUR_PATH, '../data/test_mini.csv'))
TEST_FILE_PATH = os.path.abspath(os.path.join(
    CUR_PATH, '../data/test_mini.csv'))
