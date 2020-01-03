#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ‘ÿ»Îlogger

import sys
import os
import logging
import logging.config
reload(sys)
sys.setdefaultencoding('utf-8')

cur_path = os.getcwd()#os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(cur_path, "../conf/")))

log_conf_file = os.path.join(cur_path, '../conf/log.conf')
logging.config.fileConfig(log_conf_file)
ilog = logging.getLogger('root')
ilog_info = logging.getLogger('info')
ilog_warn = logging.getLogger('warn')
