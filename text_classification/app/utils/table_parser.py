# coding: utf-8
from __future__ import unicode_literals

import re
from string_utils import count_pattern


class TableParser(object):
    """
    表格解析
    根据带表格信息的字符串解析出表格
    """

    @staticmethod
    def get_table_strings(content):
        """
        从content里面查找所有表格字符串

        :param content: string
        :type content: unicode
        :return: 
        """
        assert isinstance(content, unicode)

        regex = re.compile(r'\[\[[^\5]*\]\]', re.U)
        for idx, match in enumerate(regex.finditer(content)):
            start_idx, table_string = match.start(), match.group()
            yield idx, start_idx, table_string

    @staticmethod
    def get_pre_table_lines(table_start_idx, content, line_num=3):
        """
        获取table之前的n行内容，或遇到开头或上一个表格为止
        :param content:
        :param line_num:
        :return:
        """
        result = []
        pre_content_lines = [i for i in content[:table_start_idx-1].split('\5') if i]
        pre_content_lines.reverse()
        for line in pre_content_lines[:line_num]:
            if TableParser.judge_index_in_table(0, line):
                return result
            else:
                result.append(line)
        return result

    @staticmethod
    def get_table_row_title(table_string):
        """
        获取表格行表头
        :param table_string:
        :return:
        """
        assert isinstance(table_string, unicode)
        regex_row_title = re.compile('\[\[(.*?\])\]', re.U)
        regex_cell_content = re.compile('\[(.*?)\]', re.U)
        match = regex_row_title.search(table_string)
        start_idx, row_title = match.start(), match.group(1)
        for match in regex_cell_content.finditer(row_title):
            start_idx, row_title = match.start(), match.group(1)
            yield row_title

    @staticmethod
    def get_table_column_title(table_string):
        """
        获取表格列表头
        :param table_string:
        :return:
        """
        assert isinstance(table_string, unicode)
        regex_column_title = re.compile('\[{2,}(.*?)\]', re.U)
        for match in regex_column_title.finditer(table_string):
            start_idx, column_title = match.start(), match.group(1)
            yield column_title

    @staticmethod
    def find_cell_by_index_from_table_string(index, table_string):
        """
        根据index，从table_string拿到cell内容
        :param index:
        :type index: int
        :param table_string: 
        :type table_string: unicode
        :return: 
        """
        if index < 0:
            raise Exception('参数index应该大于等于0!'.encode('utf-8'))
        if index >= len(table_string):
            raise Exception('参数index应该小于table_string的长度!'.encode('utf-8'))
        assert isinstance(table_string, unicode)

        if table_string[index] in ['[', ']']:
            return ''
        left_index = right_index = index
        while table_string[left_index] != '[' and left_index >= 0:
            left_index -= 1
        while table_string[right_index] != ']' and right_index < len(table_string):
            right_index += 1
        cell_content = table_string[left_index + 1:right_index]
        cell_start_index = left_index + 1
        return cell_start_index, cell_content

    @staticmethod
    def get_cells_from_table_string(table_string, with_empty_cell=False):
        """
        根据table_string拿到所有的cell
        :param table_string: 
        :param with_empty_cell: 是否携带cell_content为空的结果
        :return: 
        """
        assert isinstance(table_string, unicode)

        regex = re.compile(r'\[([^\[\]]*?)\]', re.U)
        for match in regex.finditer(table_string):
            start_idx, cell_content = match.start(1), match.group(1)
            if not with_empty_cell and not cell_content:
                continue
            yield start_idx, cell_content

    @staticmethod
    def judge_index_in_table(index, content):
        """
        判断content中index位置的字符在不在表格
        :param index: 
        :type index
        :param content: 
        :type content: unicode
        :return: 
        """
        assert isinstance(content, unicode)

        for _, start_idx, table_string in TableParser.get_table_strings(content):
            end_idx = start_idx + len(table_string)
            if start_idx <= index < end_idx:
                return True
        return False

    @staticmethod
    def get_shape_from_table_string(table_string):
        """get row_num and col_num from table_string

        :param table_string: e.g. '[[[aa][ab]][[ba][bb]]]'
        :type table_string: unicode
        :return:
        """
        row_count = count_pattern(table_string, r'\]\]\[\[') + 1
        col_count = count_pattern(table_string, r'(?<!\])\]\[(?!\[)') / row_count + 1  # '][' - ']][['
        cell_count = count_pattern(table_string, r'(?<=\[)[^\[\]]*(?=\])')
        if row_count * col_count != cell_count:
            raise Exception('Extra square brackets exist!')
        else:
            return row_count, col_count
