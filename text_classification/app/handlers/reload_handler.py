# coding: utf-8
from __future__ import unicode_literals

import tornado.web
from extract_framework.models_manager.models_manager import ModelsManager

from ..driver import generate_request_id, logger_online as logger


class ReloadHandler(tornado.web.RequestHandler):
    def initialize(self, models_manager):
        if not isinstance(models_manager, ModelsManager):
            raise ValueError('参数models_manager必须是ModelsManager的实例')
        self.models_manager = models_manager

    def post(self):
        try:
            # get argument
            model_version = self.get_argument('model_version')
            caller_request_id = self.get_argument('caller_request_id', default=None)
            self_request_id = generate_request_id()
            logger.update_logger_extra(
                {'caller_request_id': caller_request_id, 'self_request_id': self_request_id}
            )
            logger.info('model_version: {}'.format(model_version))
            # 更改model软链
            logger.info('links updating ...')
            self.models_manager.update_links(model_version)
            # reload models
            logger.info('reloading ...')
            self.models_manager.reload_models()
            logger.info('reloaded')
        except Exception as e:
            logger.exception(e)
            raise Exception(e)
        finally:
            logger.update_logger_extra()
