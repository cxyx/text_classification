# coding: utf-8
from __future__ import unicode_literals

import os

import tornado
import tornado.httpclient
import tornado.ioloop
import tornado.web

# from ..driver import logger_online as logger
from ..handlers.train_predict_handlers import TrainPredictHandler
# from ..handlers.reload_handler import ReloadHandler
from ...conf import conf
from ...conf import u_shape_framework_conf


class Server(object):
    def __init__(self, output_dir, tmp_dir, upload_dir, port, model_links_dir='model', config_links_dir='config'):

        if not os.path.exists(output_dir):
            raise ValueError('参数output_dir目录并不存在'.encode('utf-8'))
        if not os.path.exists(tmp_dir):
            raise ValueError('参数tmp_dir目录并不存在'.encode('utf-8'))
        if not os.path.exists(upload_dir):
            raise ValueError('参数upload_dir目录并不存在'.encode('utf-8'))
        if not isinstance(port, int) or port <= 0 or port > 65535:
            raise ValueError('参数port必须合法(0~65535)')

        self._output_dir = output_dir
        self._tmp_dir = tmp_dir
        self._upload_dir = upload_dir
        self._port = port
        self._model_links_dir = model_links_dir
        self._config_links_dir = config_links_dir


    def start(self):
        # logger.info('start server...')
        app = tornado.web.Application(handlers=[
            (conf.PREDICT_ROUTER, TrainPredictHandler,
             {'tmp_dir': self._tmp_dir, 'upload_dir': self._upload_dir}),
        ])
        app.listen(address='0.0.0.0', port=self._port)
        # logger.info('server starts with address: 0.0.0.0, port: {}'.format(self._port))
        tornado.ioloop.IOLoop.current().start()
