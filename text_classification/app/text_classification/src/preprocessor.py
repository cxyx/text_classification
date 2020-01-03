#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from HTMLParser import HTMLParser
import chardet
import pandas as pd


from sklearn.cross_validation import train_test_split

from classify_driver import *

#sys.path.append(os.path.expanduser('~/word2vec/'))
#from stemming import Stemmer


class HTMLStripper(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return u''.join(self.fed)

    def strip_tags(self, text):
        try:
            self.reset()
            self.fed = []
            self.feed(text)
            return self.get_data()
        except Exception:
            return text

'''
class Normalizer(object):
    def __init__(self):
        self.stemmer = Stemmer()

    def normalize(self, word, ret_type='word'):
        if word.encode('utf-8').isalpha():
            return self.stemmer.stemming(word)
        elif word.isdigit():
            return 'NUM_%s' % len(word)
        elif re.search(u'\d{1,2}月\d{1,2}日|\d{1,2}月|\d{1,2}日',word):
            return 'DATE'
        elif re.search(u'\d{1,4}年',word):
            return 'YEAR'
        else:
            try:
                _ = float(word)
                return 'FLOAT'
            except ValueError:
                word = re.sub(u"[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", "",word)
                return list(word) if ret_type=='char' else word

    def text_normalize(self, word_list, ret_type='word'):
        ret_list = []
        for word in word_list:
            word_norm = self.normalize(word, ret_type)
            if type(word_norm) == type([]):
                ret_list.extend(word_norm)
            else:
                ret_list.append(word_norm)
        return ret_list
        #return [self.normalize(word) for word in word_list]
'''

class Preprocessor(object):

    def __init__(self):
        self.html_stripper_ = HTMLStripper()

    @staticmethod
    def text_decode(text):
        '''编码检测和转换'''
        if isinstance(text, unicode):
            return text
        encode_dict = chardet.detect(text)
        encoding = encode_dict['encoding']
        if encoding is None:
            encoding = 'utf-8'
        try:
            text = text.decode(encoding)
        except Exception:
            ilog_warn.warn('decode fail\t' + text + '\t' +
                           encoding + '\t' + traceback.format_exc())
            return False
        return text

    @staticmethod
    def text_transform(origin_file_path, transform_file_path, columns=None, sep='\t'):
        origin_df = pd.read_csv(open(origin_file_path, 'rU'), sep=sep,
                                header=None, index_col=None)
        origin_df.columns = columns if columns and isinstance(
            columns, type([])) else ['text', 'label']
        origin_df['item_info'] = origin_df['label'].apply( \
                lambda x: json.dumps({'label': x}))
        #origin_df['item_info'] = ''
        extra_list = [x for x in origin_df.columns if x not in ['text','label']]

        origin_df = origin_df[origin_df.label.notnull()]
        origin_df = origin_df[origin_df.text.notnull()]
        origin_df['label'] = origin_df['label'].apply(lambda x:'::'.join(x.split(' ')))
        
        #origin_df['text2'] = origin_df['text']
        origin_df.to_csv(transform_file_path,
                         Volumns=['text', 'item_info', 'label'],
                         index=None)
    
    @staticmethod
    def dataset_split(origin_file_path, transform_file_path, sep='\t', ratio=(0.1,0.0,0.9)):
        assert sum(ratio) == 1
        origin_df = pd.read_csv(open(origin_file_path, 'rU'), sep=sep,
                                header=None, index_col=None)
        origin_df = origin_df[origin_df.label.notnull()]
        origin_df = origin_df[origin_df.text.notnull()]
        #origin_df['item_info'] = ''
        origin_df['label'] = origin_df['label'].apply(lambda x:'::'.join(x.split(' ')))

        train_sz, valid_sz, test_sz = ratio
        train_df, test_df = train_test_split(
            origin_df, test_size=test_sz, random_state=1)
        
        if valid_sz > 0:
            train_df, valid_df = train_test_split(
                train_df, test_size=valid_sz/(valid_sz+train_sz), random_state=1)
            valid_df.to_csv(transform_file_path + '.valid',
                            columns=['text', 'item_info', 'label'],
                            index=None, sep=',')

        train_df.to_csv(transform_file_path + '.train',
                        columns=['text', 'item_info', 'label'],
                        index=None, sep=',')
        test_df.to_csv(transform_file_path + '.test',
                       columns=['text', 'item_info', 'label'],
                       index=None, sep=',')



def main():
    pp = Preprocessor()
    
    data_dir = '../data/test/'
    
    pp.text_transform(data_dir+'train_mini.csv',data_dir+'train_mini2.csv',columns=['label', 'text', 'item_info'], sep=',')
    #pp.dataset_split(data_dir+'data.csv',data_dir+'data.csv', sep=',')

if __name__ == "__main__":
    main()
