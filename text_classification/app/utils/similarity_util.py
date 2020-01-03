# coding: utf-8
from __future__ import unicode_literals

from collections import Counter
from collections import Iterable


class SimilarityUtil(object):
    all_similarities = ['simple_similarity', 'simple_similarity_with_count']

    @staticmethod
    def compute_simple_similarity(strings1, strings2, divide='union'):
        """
        计算简单相似度，集合交集比上并集
        :param strings1: 
        :type strings1: Iterable
        :param strings2: 
        :type strings2: Iterable
        :return: 
        :rtype float
        """
        set1, set2 = set(strings1), set(strings2)
        if len(set1 | set2) == 0:
            return 0.
        try:
            if divide == 'min_len':
                similarity = float(len(set1 & set2)) / min(len(set1), len(set2))
            else:
                similarity = float(len(set1 & set2)) / len(set1 | set2)
        except ZeroDivisionError:
            return 0.
        return similarity

    @staticmethod
    def compute_simple_similarity_with_count(strings1, strings2, threshold=1):
        """
        计算考虑了count的简单相似度
        :param strings1: 
        :type strings1: Iterable
        :param strings2: 
        :type strings2: Iterable
        :return: 
        :rtype float
        """
        set1, set2 = Counter(strings1), Counter(strings2)
        if sum((set1 | set2).values()) == 0:
            return 0.
        and_values = [v for v in (set1 & set2).values() if v >=threshold]
        union_values = [v for v in (set1 | set2).values() if v>=threshold]
        similarity = float(sum(and_values)) / sum(union_values)
        return similarity
