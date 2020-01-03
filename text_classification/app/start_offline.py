# coding: utf-8
from __future__ import unicode_literals


from .offline.service import Service
from ..conf import conf


def start():
    input_dir = 'text_classification_code/input'
    output_dir = 'text_classification_code/output'
    conf_dir = 'text_classification_code/conf'
    service = Service(input_dir, output_dir, conf_dir, conf.REDIS_HOST, conf.REDIS_PORT, conf.REDIS_DB, conf.REDIS_PWD)
    service.start()


if __name__ == '__main__':
    start()
