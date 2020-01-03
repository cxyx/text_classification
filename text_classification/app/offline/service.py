# coding: utf-8
from __future__ import unicode_literals

import u_shape_framework
from extract_framework.service.service_base import ServiceBase
from u_shape_framework.engine import initialize_engine, get_current_engine
from ..driver import logger_offline as logger
from ...conf import conf
from ...conf import u_shape_framework_conf


class Service(ServiceBase):
    def __init__(self, input_dir, output_dir, conf_dir, redis_host='127.0.0.1', redis_port=6379, redis_db=0,
                 redis_pwd=''):
        super(Service, self).__init__('text_classification_code', output_dir, redis_host, redis_port,
                                      redis_db, redis_pwd, logger)

        u_shape_framework.set_logger(logger)
        logger.info('initializing u_shape_framework engine ...')
        initialize_engine(u_shape_framework_conf)
        logger.info('engine initialize finish')

        self._input_dir = input_dir
        self._output_dir = output_dir
        self._conf_dir = conf_dir

    def _run(self, field_config):
        logger.info('get config: {}'.format(field_config.dumps()))

        # run all offline workflow
        default_workflow = conf.field_workflow_offline_config['default']
        workflow_name = conf.field_workflow_offline_config.get(field_config.field_id, default_workflow)
        workflow = get_current_engine().get_workflow(workflow_name)
        request_property = {
            "input_dir": self._input_dir,
            "output_dir": self._output_dir,
            "conf_dir": self._conf_dir,
            "field_config": field_config,
        }
        workflow.run(request_property)

        logger.info('train success')
