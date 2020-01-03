# coding: utf-8
from __future__ import unicode_literals

import abc
import codecs
import pickle
from collections import Iterable


class BaseEstimator(object):
    """
    estimator base class
    estimator的模型参数应该都在__init__函数中声明
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.model = {}

    def get_params(self):
        pass

    def set_params(self):
        pass

    @abc.abstractmethod
    def gen_features(self, documents):
        self._check_documents(documents)
        # TODO

    @abc.abstractmethod
    def fit(self, features, labels):
        """
        根据features和labels，训练模型
        模型相关请保存到self.model
        :param features: 
        :param labels:
        :return: self
        :rtype: BaseEstimator
        """
        assert len(features) == len(labels)
        # TODO

    @abc.abstractmethod
    def predict(self, features, min_confidence=0.5):
        """
        针对features，生成预测结果
        :param features:
        :param min_confidence:
        :type min_confidence: float
        :rtype: Iterable
        """
        self._check_fit()
        # TODO

    def save_model(self, model_path):
        with codecs.open(model_path, 'wb') as output:
            pickle.dump(self, output)

    @staticmethod
    def load_model(model_path):
        with codecs.open(model_path, 'rb') as input_:
            return pickle.load(input_)

    @staticmethod
    def _check_documents(documents):
        if not isinstance(documents, Iterable):
            raise ValueError('参数documents应该是Iterable实例'.encode('utf-8'))
        for document in documents:
            if not isinstance(document, unicode):
                raise ValueError('参数documents的元素应该是unicode'.encode('utf-8'))

    def _check_fit(self):
        """
        判断是否被fit
        :return: 
        """
        if not self.model:
            raise Exception('estimator并没有被fit'.encode('utf-8'))

    def __repr__(self):
        pass
