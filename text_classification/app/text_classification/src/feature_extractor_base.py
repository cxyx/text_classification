#!/usr/bin/env python
# -*- coding: utf-8 -*-


class FeatureExtractorBase(object):

    def __init__(self, module_dir):
        # 必须要接受一个输出路径，输出模型到模块路径下，防止命名冲突，建议传入调用分类器的模块路径
        pass

    def load_dict(self):
        # 装载词典，有可能包括分词词典，关键词词典，词向量模型等
        pass

    def train(self, train_file_path):
        # 训练特征抽取模型，需要训练样本，可能是统计tf、idf或者互信息量等信息
        pass

    def load_model(self):
        # 装载特征抽取模型
        pass

    def gen_feature(self, text, item_info=None):
        # 抽取特征，返回sparse matrix类型或者dense matrix类型
        pass

    def gen_feature_batch(self, text_lst, item_info_lst=None):
        # 抽取batch文本特征,
        pass

    def gen_comment(self, text, item_info=None, label=None, **kwargs):
        # 返回特征抽取备注
        pass
    
    def get_feature_len(self):
        return len(self._vectorizer.vocabulary_) if self._vectorizer else 0

    @staticmethod
    def get_text_features(df):
        # 提取dataframe中除item_info、label外的所有文本特征，返回字典
        text_col = []
        for col in df.columns:
            if col not in ('item_info', 'label') and df[col].any():
                text_col.append(col) 
        text_feature_list = []
        for index, row in df.iterrows():
            text_dict = {x:row[x].decode('utf-8') for x in text_col}
            text_feature_list.append(text_dict)
        return text_feature_list


    def text_feature_merge(self, text_features):
        # 对不同类型的文本特征进行merge
        # 每个extractor调用各自的提特征核心函数
        feature_merge = []
        for text_dict in text_features:
            merge = []
            for tag, text in text_dict.iteritems():
                merge.extend(self.gen_feature_core(text, tag))
            feature_merge.append(merge)
        return [' '.join(x) for x in feature_merge]
            
