# coding: utf-8
from __future__ import unicode_literals

import codecs
import json
import os
import itertools


def get_env(env_name, default_value, logger=None):
    if env_name not in os.environ:
        warning_meg = "can't find env: {}, use default: {}".format(env_name, default_value)
        if logger:
            logger.warning(warning_meg)
        else:
            print(warning_meg)
        return default_value
    return os.environ[env_name]


def flat_nested_list(nested_list, flat_depth=1):
    """flat nested list
    :param nested_list: nested list
    :type nested_list: list
    :param flat_depth: depth of iterative to flat list
    :type flat_depth: int
    :return: list which nest level = nested_list - flat_depth
    :rtype: list
    """
    tmp_lst = nested_list
    for round in range(flat_depth):
        tmp_lst = list(itertools.chain(*tmp_lst))
    return tmp_lst


def judge_label_error(table_y_list, documents, tagged_data_json_path, field_id):
    class_one_num = 0
    class_zero_num = 0
    for table_y in table_y_list:
        if table_y:
            class_one_num += 1
        else:
            class_zero_num += 1

    if class_zero_num == 0 or class_one_num == 0:
        error_doc_ids = []
        error_label_items = []
        tagged_data_list = json.load(codecs.open(tagged_data_json_path, encoding='utf-8'))
        for document in documents:
            for tagged_data in tagged_data_list:
                content = tagged_data['content']
                if document == content:
                    error_doc_ids.append(tagged_data['doc_id'])
                    error_label_items.append(
                        [label_item for label_item in tagged_data['fields'] if label_item['field_id'] == field_id])
        error_message = 'one class exception , and the possible table label error is:\t error_doc_ids: {}\t error_label_items: {}'.format(
            json.dumps(error_doc_ids), json.dumps(error_label_items, ensure_ascii=False))
        return True, error_message
    else:
        return False, 'not label error but other error leading to svm fit exception'
