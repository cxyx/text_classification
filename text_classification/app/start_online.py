# coding: utf-8
from __future__ import unicode_literals

import sys
from .online.server import Server
from ..conf import conf



def start():
    output_dir = 'text_classification_code/output'
    tmp_dir = 'text_classification_code/tmp'
    upload_dir = 'text_classification_code/upload'
    port = conf.RECV_PORT
    model_links_dir = 'model'
    config_links_dir = 'config'

    server = Server(output_dir, tmp_dir, upload_dir, port, model_links_dir, config_links_dir)
    server.start()


if __name__ == '__main__':
    start()
