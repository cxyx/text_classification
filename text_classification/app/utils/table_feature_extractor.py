# -*- coding: utf-8 -*-
# email: qianyixin@datagrand.com
# date: 2019-07-09 01:21
from __future__ import unicode_literals

from .. import driver
from table_parser import TableParser
from feature_util import get_ner_type


class TableFeatureExtractor(object):
    """Class for extract specific table features"""
    all_features = ['pre_table_lines', 'label_table_indices', 'avg_tagged_cell_count', 'table_row_titles',
                    'table_column_titles', 'tagged_cells', 'table_all_cells']

    @staticmethod
    def extract_pre_table_lines(table_info, line_num=3):
        """generate line contents before single table

        :param table_info: dict contains table meta features, keys are feature name, values are features
        :type table_info: dict
        :param line_num: num of line contents before each table to extract
        :type line_num: int
        :return: element is line content previous to each table within doc
        :rtype: list[str]
        """
        assert 'document' in table_info, 'document missing in input features'
        assert 'table_start_idx' in table_info, 'table_start_indices missing in input features'
        document = table_info['document']
        table_start_idx = table_info['table_start_idx']
        pre_table_lines = TableParser.get_pre_table_lines(table_start_idx, document, line_num)
        return pre_table_lines

    @staticmethod
    def extract_label_table_indices(table_info, table_count):
        """generate relative idx of labeled table within doc

        :param table_info: dict contains table meta features, keys are feature name, values are features
        :type table_info: dict
        :param table_count: total num of tables within a doc
        :type table_count: int
        :return: idx relative to all table
        :rtype: int
        """
        assert 'table_idx' in table_info, 'table_idx missing in input features'
        table_idx = table_info['table_idx']
        label_table_indices = float(table_idx) / table_count
        return label_table_indices

    @staticmethod
    def extract_tagged_cells(table_info, table_label):
        """extract contents of tagged cells single of labeled table

        :param table_info: dict contains table meta features, keys are feature name, values are features
        :type table_info: dict
        :param table_label: element is start and end idx of each label within doc
        :type table_label: list[list]
        :return: first list contain content of tagged cell within table, second is num of tagged cells
        :rtype: list[str]
        """
        assert 'table_string' in table_info, 'table_string missing in input features'
        table_string = table_info['table_string']
        tagged_cell_contents = []
        for start_idx, end_idx in table_label:
            _, tagged_cell_content = TableParser.find_cell_by_index_from_table_string(start_idx, table_string)
            tagged_cell_contents.append(tagged_cell_content)
        return tagged_cell_contents

    @staticmethod
    def extract_tagged_cell_count(table_label):
        """generate count of tagged cells within single labeled table

        :param table_label: element is start and end idx of each label within doc
        :type table_label: list[list]
        :return: num of tagged cells
        :rtype: int
        """
        return len(table_label)

    @staticmethod
    def extract_table_cells(table_info):
        """extract contents of all cells of single labeled table

        :param table_info: dict contains table meta features, keys are feature name, values are features
        :type table_info: dict
        :return: first contains relative idx of cells within table, second contains all cells' contents
        :rtype: tuple[list[str], int]
        """
        assert 'table_string' in table_info, 'table_string missing in input features'
        table_string = table_info['table_string']
        table_cells = []
        table_cell_start_indices = []
        for start_idx, cell_content in TableParser.get_cells_from_table_string(table_string):
            table_cells.append(cell_content)
            table_cell_start_indices.append(start_idx)
        return table_cell_start_indices, table_cells

    @staticmethod
    def extract_table_row_titles(table_info):
        """extract content of row titles of single labeled table

        :param table_info: dict contains table meta features, keys are feature name, values are features
        :type table_info: dict
        :return: each element is row title contents of single table, length equals column num of table
        :rtype: list[str]
        """
        assert 'table_string' in table_info, 'table_string missing in input features'
        table_string = table_info['table_string']
        table_row_titles = list(TableParser.get_table_row_title(table_string))
        return table_row_titles

    @staticmethod
    def extract_table_column_titles(table_info):
        """extract content of column titles of single labeled table

        :param table_info: dict contains table meta features, keys are feature name, values are features
        :type table_info: dict
        :return: each element is column title contents of single table, length equals row num of table
        :rtype: list[str]
        """
        assert 'table_string' in table_info, 'table_string missing in input features'
        table_string = table_info['table_string']
        table_column_titles = list(TableParser.get_table_column_title(table_string))
        return table_column_titles

    @staticmethod
    def norm_feature(table_feature, ner_obj=None, ner_types=None):
        """normalize text into ner tag

        :param table_feature: raw feature values extract from table to be normed
        :type table_feature: list[str]
        :param ner_obj: object to convert specific text to ner tags
        :type ner_obj: object
        :param ner_types: define which type of ner tags text would be normalized, default contains all kinds of types
        :type ner_types: list[str]
        :return: features values after normalized
        :rtype: list[str]
        """
        if not ner_obj:
            ner_obj = driver.ner
        if not ner_types:
            ner_types = ["person", "org", "location", "time"]
        normed_table_feature = []
        ner_counts = {ner_type: 0 for ner_type in ner_types}
        for feature_content in table_feature:
            feature_ner_type = get_ner_type(feature_content, ner_obj)
            if feature_ner_type:
                ner_counts[feature_ner_type] += 1
                feature_content = '{}_{}'.format(feature_ner_type, ner_counts[feature_ner_type])
            normed_table_feature.append(feature_content)
        return normed_table_feature

    def process_tagged_doc(self, tagged_docs, labels, feature_names):
        """method to extract specified table features from tagged docs

        :param tagged_docs: contains meta data of tagged docs, e.g. doc content, tables' meta data, etc
        :type tagged_docs: dict
        :param labels: inner values are start and end idx of labeled.
                       first dimension is table num within doc, second is num of labeled cells within table.
        :type labels: list[list[int]]
        :param feature_names: define which type of feature should be extract from input data
        :type feature_names: list[str]
        :return: keys are feature names, values are feature value lists
        :rtype: dict[str, list]
        """
        # load input meta feature
        documents = tagged_docs['documents']
        table_indices = tagged_docs['table_indices']
        table_start_indices = tagged_docs['table_start_indices']
        table_strings = tagged_docs['table_strings']
        doc_indices = tagged_docs['doc_indices']
        # init containers and check validation of feature names
        feature_values = {}
        for feature_name in feature_names:
            feature_values[feature_name] = []
        # add container for cell_index relative to table
        if 'table_all_cells' in feature_values:
            feature_values['table_cell_start_indices'] = []
        # extract features for each table
        for idx, doc_idx in enumerate(doc_indices):
            table_info = {
                'doc_idx': doc_idx,
                'document': documents[doc_idx],  # get corresponding doc
                'table_idx': table_indices[idx],
                'table_start_idx': table_start_indices[idx],
                'table_string': table_strings[idx],
            }
            table_label = labels[idx]
            table_count = len(list(TableParser.get_table_strings(documents[doc_idx])))
            if 'pre_table_lines' in feature_names:
                feature_values['pre_table_lines'].append(self.extract_pre_table_lines(table_info))
            if 'table_row_titles' in feature_names:
                feature_values['table_row_titles'].append(self.extract_table_row_titles(table_info))
            if 'table_column_titles' in feature_names:
                feature_values['table_column_titles'].append(self.extract_table_column_titles(table_info))
            if 'label_table_indices' in feature_names:
                feature_values['label_table_indices'].append(self.extract_label_table_indices(table_info, table_count))
            if 'tagged_cells' in feature_names:
                feature_values['tagged_cells'].append(self.extract_tagged_cells(table_info, table_label))
            if 'avg_tagged_cell_count' in feature_names:
                feature_values['avg_tagged_cell_count'].append(self.extract_tagged_cell_count(table_label))
            if 'table_all_cells' in feature_names:
                tmp_cell_start_indices, tmp_all_cells = self.extract_table_cells(table_info)
                feature_values['table_cell_start_indices'].append(tmp_cell_start_indices)
                feature_values['table_all_cells'].append(tmp_all_cells)

        return feature_values

    def process_untagged_doc(self, untagged_doc, feature_names):
        """method to extract specified table features from untagged docs

        :param untagged_doc: contains meta data of tagged docs, e.g. doc content, tables' meta data, etc
        :type untagged_doc: dict
        :param feature_names: define which type of feature should be extract from input data,
                              only those non-label-based features are support
        :type feature_names: list[str]
        :return: keys are feature names, values are feature value lists
        :rtype: dict
        """
        # load input meta feature
        document = untagged_doc['document']
        table_start_indices = untagged_doc['table_start_indices']
        table_strings = untagged_doc['table_strings']
        # init containers
        feature_values = {}
        for feature_name in feature_names:
            feature_values[feature_name] = []
        # add container for cell_index relative to table
        feature_values['table_cell_start_indices'] = []
        # generate feature for each table within single doc
        for table_start_idx, table_string in zip(table_start_indices, table_strings):
            table_info = {
                'document': document,
                'table_start_idx': table_start_idx,
                'table_string': table_string,
            }
            if 'pre_table_lines' in feature_names:
                feature_values['pre_table_lines'].append(self.extract_pre_table_lines(table_info))
            if 'table_row_titles' in feature_names:
                feature_values['table_row_titles'].append(self.extract_table_row_titles(table_info))
            if 'table_column_titles' in feature_names:
                feature_values['table_column_titles'].append(self.extract_table_column_titles(table_info))
            if 'table_all_cells' in feature_names:
                tmp_cell_start_indices, tmp_all_cells = self.extract_table_cells(table_info)
                feature_values['table_cell_start_indices'].append(tmp_cell_start_indices)
                feature_values['table_all_cells'].append(tmp_all_cells)

        return feature_values
