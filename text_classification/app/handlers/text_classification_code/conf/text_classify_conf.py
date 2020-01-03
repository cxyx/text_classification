#! /usr/bin/env python
#coding:utf-8

import os
cur_path = os.path.join(os.getcwd(), '../conf/')#os.path.dirname(__file__)


TRAIN_FILE_PATH = os.path.abspath(os.path.join(
    cur_path, '../data/demo/test_mini.csv'))
TEST_FILE_PATH = os.path.abspath(os.path.join(
    cur_path, '../data/demo/test_mini.csv'))

'''logging file path'''
LOG_CONF_PATH = os.path.abspath(os.path.join(cur_path, 'log.conf'))

'''check online'''
ENV_STR = 'development'
if 'ENVIRONMENT' in os.environ:
    ENV_STR = os.environ['ENVIRONMENT']

'''tornado参数'''
RECV_IP = '0.0.0.0'
RECV_PORT = 9760
PROCESS_NUM = 1
if ENV_STR=='production':
    # RECV_IP = '127.0.0.1'
    PROCESS_NUM = 4
PATH_PREFIX = '/classify/'
AUDIT_PREFIX = '/audit/'
COMMENT_PREFIX = '/bad_comment/'

'''mysql configuration'''
MYSQL_SERVER = 'rds77iv9etcjpi86u2895.mysql.rds.aliyuncs.com'
MYSQL_USER = 'data_grand'
MYSQL_PASSWORD = '23Knowledge'
if ENV_STR == 'production':
    MYSQL_SERVER = 'rdsq3zsso4e737w4gwjq.mysql.rds.aliyuncs.com'
    MYSQL_USER = 'siterec'
    MYSQL_PASSWORD = 'siterec123456'
MYSQL_PORT = 3306
DATACENTER_DB = 'siterec_datacenter'
DASHBOARD_DB = 'siterec_dashboard'

'''scribe configuration'''
SCRIBE_SERVER_IP = '127.0.0.1'
SCRIBE_SERVER_PORT = 1463
SCRIBE_CATEGORY = 'text_mining'

STATUS_OK = 'OK'
STATUS_WARN = 'WARN'
STATUS_FAIL = 'FAIL'

URL_PATTERN = ur'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?'
