# coding:utf-8
# @email: chenyu@datagrand.com
# @author: chenyu

import codecs
import json
import sys

from pdf2txt_decoder.pdf2txt_decoder import Pdf2TxtDecoder

reload(sys)
sys.setdefaultencoding('utf-8')


def load_data(json_file):
    with codecs.open(json_file, 'r', encoding='utf-8') as f:
        content = json.load(f, encoding='utf-8')
    content.update({
        'pdf2txt_decoder':
        # Pdf2TxtDecoder(content.get('rich_content', {})),
        Pdf2TxtDecoder(content),
    })
    return content


def dump_data(data, name, keys=('rich_content', 'doc_type')):
    pass
