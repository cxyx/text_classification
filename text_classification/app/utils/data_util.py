# coding: utf-8
from __future__ import unicode_literals

import codecs
import re
import math

from .table_parser import TableParser
from .string_utils import count_pattern


def parse_tagged_data(data_path):
    """
    解析\3\4标注数据
    :param data_path: 
    :return: 
    """
    for line in codecs.open(data_path, 'r', 'utf-8'):
        line = line.strip()
        if line:
            content = ''
            label_indices = []
            for term in line.split('\3'):
                value, tag = term.split('\4')
                content += value
                if tag != 'O':
                    start_idx = content.rindex(value)
                    end_idx = start_idx + len(value)
                    label_indices.append([start_idx, end_idx])
            yield content, label_indices


def transform_document_label_to_table_label(document, label, with_empty_label=False):
    """
    将文档标注转换为表格标注, 一个文档可能包含多个表格

    :param document: 
    :type document: unicode
    :param label: list[list[int]], element contains start and end idx of label, e.g. [[1, 3], [5, 7]]
    :type label: list
    :param with_empty_label 是否过滤标注为空的表格
    :type with_empty_label: bool
    :return: 
    """
    for table_idx, table_start_idx, table_string in TableParser.get_table_strings(document):
        table_end_idx = table_start_idx + len(table_string)
        table_label = []  # 当前表格的label
        for label_start_idx, label_end_idx in label:
            if table_start_idx <= label_start_idx < table_end_idx and \
                    table_start_idx <= label_end_idx - 1 < table_end_idx:  # 标注在当前表格里面
                table_label.append([label_start_idx - table_start_idx, label_end_idx - table_start_idx])
        if not with_empty_label and not table_label:  # 当前表格没标注数据
            continue
        yield table_idx, table_start_idx, table_string, table_label


def transform_table_label_to_array(table_string, table_shape, table_label=None):
    """
    transform table string and table label into two-dimension numpy array

    :param table_string: e.g. '[[[aa][ab]][[ba][bb]]]'
    :type table_string: unicode
    :param table_shape: element is row_num and col_num of table_string
    :type table_shape: tuple(int)
    :param table_label: element is relative start and end index of label within table
    :type table_label: list[list[int]]
    :return:
    """
    if table_label is None:
        table_label = []
    row_count, col_count = table_shape
    table_array = [[0 for _ in range(col_count)] for _ in range(row_count)]
    labels_list = []  # list to contain label idx within table array
    cell_pattern = r'(?<=\[)[^\[\]]*(?=\])'  # most inside contents that bound with [] a.k.a. cell content
    for idx, m in enumerate(re.finditer(cell_pattern, table_string)):
        cell_start_idx = m.start()
        cell_end_idx = m.end()
        cell_content = m.group()
        cell_row_idx = int(math.floor(idx / col_count))
        cell_col_idx = int(idx % col_count)
        cell_label = 1 if [cell_start_idx, cell_end_idx] in table_label else 0
        table_array[cell_row_idx][cell_col_idx] = cell_content
        labels_list.append(cell_label)

    return table_array, labels_list
