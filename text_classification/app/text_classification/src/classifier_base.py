#!/usr/bin/env python
# coding: utf-8

import os
import operator
from collections import defaultdict
import pandas as pd
import numpy as np
import math
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split, KFold
from classify_driver import *
from feature_extractor_base import FeatureExtractorBase


class ClassifierBase(object):
    """main comment process for porn"""

    def __init__(self, train_file_path=None,
                 module_dir=None):
        # 其中 train_file_path是训练集路径，默认值见配置
        # 其中 module_dir是模块输出路径，输出模型到模块路径下，防止文件名冲突，默认一个分类器建立一个目录

        if not train_file_path:
            self._train_file_path = tc_conf.TRAIN_FILE_PATH
        else:
            self._train_file_path = os.path.abspath(train_file_path)

        self._classifier_name = self.__class__.__name__.lower().replace('classifier', '')
        if not module_dir:
            self._module_dir = os.path.abspath(os.path.join(
                cur_path, "../data/%s/" % self._classifier_name))
        else:
            self._module_dir = os.path.abspath(module_dir)

        if not os.path.exists(self._module_dir):
            os.mkdir(self._module_dir)

        self._feature_extractor = None
        self._model_args = {}

    def set_file_path(self, train_file_path=None, module_dir=None):
        if train_file_path:
            self._train_file_path = train_file_path

        if module_dir:
            self._module_dir = module_dir

    def set_model_args(self, args):
        if args:
            self._model_args = args

    def train(self):
        # 训练模型
        pass

    def check_status(self):
        return bool(self._model)

    def cross_validation(self, k_fold=3, fixed=True):
        df = pd.read_csv(
            open(self._train_file_path), index_col=None)
        
        #train_df, test_df = train_test_split(
        #    df, train_size=train_size, random_state=random_state)
        
        headers = ["precision", "recall", "f1-score", "support"]
        cv_statistics = {}
        cv_dir = self._module_dir + '/cv'
        if not os.path.exists(cv_dir):
            os.makedirs(cv_dir)
        for fname in os.listdir(cv_dir):
            os.remove(os.path.join(cv_dir, fname))

        rs = 2 if fixed else None
        kf = KFold(n=len(df), n_folds=k_fold, shuffle=True, random_state=rs)
        k = 0
        for train_index, test_index in kf:
            train_df = df.iloc[train_index]
            test_df = df.iloc[test_index]

            origin_train_file_path = self._train_file_path
            
            self._train_file_path = cv_dir + '/cv_fold_%d.train' % k
            train_df.to_csv(self._train_file_path, index=None)
            self.train(k_fold=False)

            test_file_path = cv_dir + '/cv_fold_%d.test' % k
            test_df.to_csv(test_file_path, index=None)

            self._train_file_path = origin_train_file_path

            predict_file_path = cv_dir + '/cv_fold_%d.pred' % k
            self.predict(test_file_path, predict_file_path)

            diff_file_path = cv_dir + '/cv_fold_%d.diff' % k
            evaluation_file_path = cv_dir + '/cv.eval'
            report_dict = self.evaluation(test_file_path, predict_file_path,
                                diff_file_path, evaluation_file_path, print_report=False)
            for cate in report_dict:
                cv_statistics.setdefault(cate,{h:0.0 for h in headers})
                cv_statistics[cate]['support'] += report_dict[cate]['support']
                for metric in headers[:-1]:
                    cv_statistics[cate][metric] += report_dict[cate][metric] * report_dict[cate]['support']
            k += 1

        title = u'%d-Fold' % k_fold
        width = max(max(len(nm) for nm in report_dict), len('avg / total') ,len(title)) + 5
        head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
        cv_report = head_fmt.format(title, *headers, width=width)
        cv_report += u'\n\n'
        row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'
        for cate in report_dict:
            row = [cate]
            for metric in headers[:-1]:
                cv_statistics[cate][metric] /= cv_statistics[cate]['support']
                row.append(cv_statistics[cate][metric])
            row.append(int(cv_statistics[cate]['support']))
            row_str = row_fmt.format(*row, width=width, digits=4)
            if cate == u'avg / total':
                avg_str = row_str
                continue
            cv_report += row_str
        cv_report += u'\n' + avg_str
        print cv_report
        ClassifierBase.save_report_dict(cv_report, self._module_dir+'/cv_eval.txt')

    def load_model(self):
        # 装载模型
        # self._model = joblib.load(self._model_file_path)
        # if self._feature_extractor:
        #     self._feature_extractor.load_model()
        pass

    def train_prepare(self):
        # 返回模型训练所需的特征和标签
        self._feature_extractor.train(self._train_file_path)

        train_df = pd.read_csv(open(self._train_file_path, 'rU'),
                               index_col=None)
        text_features = FeatureExtractorBase.get_text_features(train_df)
        
        item_info = train_df['item_info'] if 'item_info' in train_df and \
            train_df['item_info'].any() else None
        X = self._feature_extractor.gen_feature_batch(text_features, item_info)
        return X, train_df['label']

    def speed_test(self, df):
        # 测试预测速度
        test_size = 100
        st_time = time.time()
        for text in df['text'].apply(str).values[:test_size]:
            self.classify(text)
        print 'Average predict cost %.04f ms.' % ((time.time()-st_time)/test_size*1000)
   

    def predict(self, test_file_path, predict_file_path=None,
                multi_label=False, weight_threshold=0.5):

        if not predict_file_path:
            predict_file_path = self._module_dir + '/predict.csv'

        test_df = pd.read_csv(open(test_file_path, 'rU'), dtype=object, index_col=None)
        test_df.fillna('', inplace=True)

        test_df['comment'] = ''
        
        # self.speed_test(test_df)

        text_feature_batch = FeatureExtractorBase.get_text_features(test_df)
        
        item_info = test_df.get('item_info')
        predict_list = self.classify_batch(text_feature_batch, item_info)

        multi_pred = lambda x: '::'.join([k for k,v in x.iteritems() if v>weight_threshold])
        max_pred = lambda x: max(x.iteritems(), key=operator.itemgetter(1))[0]
        pred_func = lambda x: multi_pred(x) or max_pred(x) if multi_label else max_pred(x)
        predict_label = pd.Series(predict_list).apply(pred_func)
        
        test_df['predict_label'] = predict_label
        
        for index, text_feature in enumerate(text_feature_batch):
            item_info = test_df.iloc[index].get('item_info')
            pred = test_df.iloc[index]['predict_label']
            comment_ret = self.gen_comment(text_feature, item_info, pred)
            if comment_ret:
                test_df.set_value(index, 'comment', comment_ret)

        test_df.to_csv(predict_file_path, index=None)
    
    @staticmethod
    def save_report_dict(report, evaluation_file_path):
        with open(evaluation_file_path, 'w') as f:
            f.write(report)

    @staticmethod
    def report2dict(cr):
        # Parse rows
        tmp = list()
        for row in cr.split("\n"):
            parsed_row = [x.strip() for x in row.split("  ") if len(x) > 0]
            if len(parsed_row) > 0:
                tmp.append(parsed_row)

        # Store in dictionary
        measures = tmp[0]

        D_class_data = defaultdict(dict)
        for row in tmp[1:]:
            class_label = row[0]
            for j, m in enumerate(measures):
                D_class_data[class_label][
                    m.strip()] = float(row[j + 1].strip())
        return D_class_data

    @staticmethod
    def get_top_k_pred(pred_ret, k=1):
        return sorted(pred_ret.items(), key=lambda x:x[1], reverse=True)[:k]

    @staticmethod
    def result_merge(ret_list):
        sigmoid = lambda x: 1/(1+math.exp(-2*x))
        def merge(ret_A, ret_B):
            for label in ret_B:
                ret_A[label] = ret_A.get(label, 0.0) + ret_B[label]
                # 单调平滑归一
                if ret_A[label] > 0.843:
                    ret_A[label] = sigmoid(ret_A[label])
            return ret_A
        return reduce(merge, ret_list)

    @staticmethod
    def badcase_static(report_dict):
        precision_dict = {}
        recall_dict = {}

        for cate_name in report_dict:
            if cate_name.strip() == 'avg / total':
                continue

            precision_dict[cate_name] = report_dict[cate_name]['precision']
            recall_dict[cate_name] = report_dict[cate_name]['recall']

        top_5_precision_lst = []
        for cate_name, precision in sorted(precision_dict.items(), key=operator.itemgetter(1))[:5]:
            top_5_precision_lst.append((cate_name, precision))

        top_5_recall_lst = []
        for cate_name, recall in sorted(recall_dict.items(), key=operator.itemgetter(1))[:5]:
            top_5_recall_lst.append((cate_name, recall))

        print 'top5 precision:\t%s\ntop5 recall:\t%s' % (
            '\t'.join([str(cate_name) + ":" + str(precision)
                       for (cate_name, precision) in top_5_precision_lst]),
            '\t'.join([str(cate_name) + ":" + str(recall)
                       for (cate_name, recall) in top_5_recall_lst])
        )
    
    #@timing
    def evaluation(self, test_file_path, predict_file_path=None, diff_file_path=None, 
                          evaluation_file_path=None, print_report=True):
        # 评估结果，输出全量结果到predict_file_path，输出diff结果到diff_file_path

        if not predict_file_path:
            predict_file_path = self._module_dir + '/predict.csv'

        if not diff_file_path:
            diff_file_path = self._module_dir + '/diff.csv'

        if not evaluation_file_path:
            evaluation_file_path = self._module_dir + '/evaluation.txt'

        test_df = pd.read_csv(open(test_file_path, 'rU'), dtype=object, index_col=None,
                              usecols=['label', 'text', 'item_info'])

        # 评估结果
        predict_df = pd.read_csv(open(predict_file_path, 'rU'), dtype=object, index_col=None,
                                 usecols=['predict_label', 'comment'])

        le = LabelEncoder()

        test_label_list = list(test_df['label'].unique())
        predict_label_list = list(predict_df['predict_label'].unique())
        total_label_list = []
        for multi_label in test_label_list + predict_label_list:
            try:
                label_lst = multi_label.split('::')
                total_label_list.extend(label_lst)
            except:
                print multi_label
                raise
        total_label_list = list(set(total_label_list))

        le.fit(total_label_list)

        # 生成全量结果分析统计
        y_true_matrix = []
        y_pred_matrix = []

        le_len = len(le.classes_)

        for labels_true in test_df['label'].str.split('::'):
            y_true = le.transform(labels_true)
            y_vec = [0] * le_len
            for y in y_true:
                y_vec[y] = 1
            y_true_matrix.append(y_vec)

        for labels_pred in predict_df['predict_label'].str.split('::'):
            y_pred = le.transform(labels_pred)
            y_vec = [0] * le_len
            for y in y_pred:
                y_vec[y] = 1
            y_pred_matrix.append(y_vec)
        classify_result = classification_report(
            np.array(y_true_matrix), np.array(y_pred_matrix), target_names=le.classes_, digits=4)
        
        #ilog_info.info('%s classification report:\n%s',
        #               self._classifier_name, classify_result)
        report_dict = ClassifierBase.report2dict(classify_result)
        if print_report:
            print classify_result
            self.save_report_dict(classify_result, evaluation_file_path)
            self.badcase_static(report_dict)

        # 生成diff结果
        merge_df = pd.concat(
            [test_df, predict_df['predict_label'], predict_df['comment']], axis=1)
        diff_df = merge_df.loc[merge_df['label'] != merge_df['predict_label']]
        del diff_df['comment']
        diff_df.to_csv(diff_file_path, index=None)
        return report_dict

    def classify(self, text_features, item_info=None):
        # 输出分类结果
        # 分类结果的类型为dict类型，形式为{cate_name1:score1,cate_name2:score2}
        pass
    
    def classify_comment(self, text_features, item_info=None):
        return self.classify(text_features, item_info), ''

    def classify_batch(self, text_batch, item_info_batch=None, regex_rule=False):
        # 批量分类，提升速度需重写为predict_proba
        return [self.classify(x) for x in text_batch]
        

    def gen_comment(self, text_features, item_info=None, label=None):
        # 生成分类备注信息，譬如可以输出 命中的规则，或者说抽取到的特征
        pass
