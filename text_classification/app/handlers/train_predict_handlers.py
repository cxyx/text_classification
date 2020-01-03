# coding=utf-8
from __future__ import unicode_literals

import codecs
import json
import time

import tornado.web
# from ..driver import generate_request_id, logger_online as logger
import os

from .text_classification_code.src.classify_driver import *
from .text_classification_code.src.config_classifier import ConfigClassifier


class TrainPredictHandler(tornado.web.RequestHandler):

    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def post(self):
        data = json.loads(self.request.body)
        init_time = time.time()

        module_dir = data.get('project_dir', None)
        if not os.path.exists(module_dir):
            print '项目文件夹%s不存在，请创建!' % module_dir
            exit(1)

        cls = ConfigClassifier(module_dir = module_dir)
        if data.get('train', None):
            train_path = os.path.join(module_dir, data.get('train', None))
            if not os.path.exists(module_dir):
                print '找不到训练文件: %s' % train_path
                exit(1)

            cls.set_file_path(train_file_path = train_path)
            cls.train(k_fold = data.get('k_fold', None))

        if data.get('predict', None):
            test_path = os.path.join(module_dir, data.get('predict', None))
            if not os.path.exists(module_dir):
                print '找不到测试或待预测文件: %s' % test_path
                exit(1)
            cls.load_model()
            cls.predict(test_path)

            if data.get('evaluation', None):
                cls.evaluation(test_path)



