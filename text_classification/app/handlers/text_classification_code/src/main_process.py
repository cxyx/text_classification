#!/usr/bin/env python
#coding: utf-8

from classify_driver import *
from logger import *
from tornado import httpserver
from tornado import ioloop
from classify_operator import ClassifyOperator

import argparse
parser = argparse.ArgumentParser(description='指定项目名，启动分类模块HTTP服务')

parser.add_argument('project_name', nargs='*', help='项目文件夹名，一般位于../data/目录下, 支持多个')
parser.add_argument('-p','--port', type=int, nargs='?', default=9760, help='分类服务的端口号, 默认9760')
parser.add_argument('-n','--process_num', type=int, nargs='?', default=4, help='分类服务的进程数，0表示核心数，默认4')
args = parser.parse_args()

g_classifier = ClassifyOperator(args.project_name)

def init_global():
    global g_classifier
    g_classifier = ClassifyOperator()


def parse_input(args):
    text_dict = {k:v[0] for k,v in args.items() if k!='item_info'}
    item_info = args['item_info'][0] if args.get('item_info') else ''
    return text_dict, item_info


def handle_request(request):

    try:
        global g_classifier
        if g_classifier is None:
            init_global()
        start_time = time.time()
        project_name = request.path.strip('/')
        ip = request.headers.get("X-Real-Ip",'')
        if ip == "": ip = request.remote_ip
        text_dict, item_info = parse_input(request.arguments)
        ret = g_classifier.text_classify(project_name, ip, text_dict, item_info)
        message = json.dumps(ret,ensure_ascii=False).encode('utf-8')
        request.write("HTTP/1.1 200 OK\r\nContent-Length: %d\r\nContent-Type: text/HTML\r\n\r\n%s" \
            % (len(message), message))

        pro_time = (time.time() - start_time) * 1000
        ilog_info.info('process time\t' + str(pro_time) + ' ms')
    except Exception, e:
        tb = traceback.format_exc()
        ilog.error('handle_request_failed\t%s\t%s\t%s' %(tb,request.arguments,str(e)))
        message = '请求错误'
        request.write("HTTP/1.1 400 Bad request\r\nContent-Length: %d\r\nContent-Type: text/HTML\r\n\r\n%s" \
            % (len(message), message))

    request.finish()


def main():
    try:
        ilog_info.info('start text_classify...')
        http_server = httpserver.HTTPServer(handle_request)
        http_server.bind(args.port)
        http_server.start(args.process_num)
        ilog_info.info('start service...')
        ioloop.IOLoop.instance().start()
    except Exception as e:
        tb = traceback.format_exc()
        ilog.error('text_classify failed: %s %s' %(tb,str(e)))


if __name__ == '__main__':
    main()
