#!/usr/bin/env python 
# coding:utf-8

from classify_driver import *
from classifier_base import ClassifierBase

# feature extractor
from feature_extractor_base import FeatureExtractorBase
from ngram_extractor import NgramExtractor
from wordseg_extractor import WordsegExtractor
from embedding_extractor import EmbeddingExtractor
from skipgram_extractor import SkipgramExtractor
from merge_extractor import MergeExtractor

# classifier
#from xgboost_basic_classifier import XgboostBasicClassifier
from svm_basic_classifier import SvmBasicClassifier
from svm_rule_classifier import SvmRuleClassifier
from lr_basic_classifier import LrBasicClassifier
from lr_rule_classifier import LrRuleClassifier
from multi_classifier import MultiClassifier
from multi_rule_classifier import MultiRuleClassifier


CLASSIFIER_MAP = {
    #'xgboost':XgboostBasicClassifier,
    'lr':LrBasicClassifier,
    'lr_rule': LrRuleClassifier,
    'svm':SvmBasicClassifier,
    'svm_rule':SvmRuleClassifier,
    'multi':MultiClassifier,
    'multi_rule':MultiRuleClassifier
}

class ConfigClassifier(ClassifierBase):
    '''
    功能：读取配置信息来进行模型训练
    '''
    def __init__(self, module_dir, train_file_path=None):
        ClassifierBase.__init__(self, train_file_path, module_dir)
        self._parse_args()
        self._init_object()
        print '---> init done <---'

    def _init_object(self):
        try:
            self._classifier = CLASSIFIER_MAP[self._classifier_name]\
                                (self._train_file_path, self._module_dir)
            self._feature_extractor = MergeExtractor(self._module_dir)
            self._feature_extractor.set_configuration(self._feature_method, self._feature_args)
            self._classifier.set_feature_extractor(self._feature_extractor)
            self._classifier.set_model_args(self._classifier_args)
        except Exception as e:
            print 'init object failed.\t', traceback.format_exc()
        return

    def _parse_args(self):
        try:
            # 优先读取module_dir下的params_dict
            sys.path.insert(0, os.path.abspath(self._module_dir))
            from params_conf import params_dict
            #print params_dict
            self._feature_method = params_dict.get('feature_method')
            self._feature_args = params_dict.get('feature_args')
            self._classifier_name = params_dict.get('classifier')
            self._classifier_args = params_dict.get('classifier_args',{})
        except Exception as e:
            print 'parser args fail', traceback.format_exc()
        return

    
    def set_file_path(self, train_file_path=None, module_dir=None):
        if train_file_path:
            self._train_file_path = train_file_path
        if module_dir:
            self._module_dir = module_dir
        self._classifier.set_file_path(train_file_path, module_dir)

    def load_model(self):
        self._classifier.load_model()
        self._feature_extractor.load_model()
        print '---> model loaded <---'
        print 'number of features:', self._feature_extractor.get_feature_len()

    def train(self, k_fold=False):
        try:
            self._feature_extractor.train(self._train_file_path)
            self._classifier.train(k_fold=k_fold)
            print '---> training complete <---'
            print 'number of features:', self._feature_extractor.get_feature_len()
        except Exception as e:
            print 'train model failed.', traceback.format_exc()

    def classify(self, text_dict, item_info=''):
        return self._classifier.classify(text_dict, item_info)
    
    def classify_single(self, text_dict, item_info=''):
        multi_pred = lambda x: sorted([(k,v) for k,v in x.iteritems() if v>0.5],
                        key=lambda y:y[1], reverse=True)
        max_pred = lambda x: max(x.iteritems(), key=lambda y:y[1])
        pred_func = lambda x: multi_pred(x) or max_pred(x) \
            if 'multi' in self._classifier.__class__.__name__ else max_pred(x)
        return pred_func(self.classify(text_dict, item_info))

    def classify_batch(self, text_batch, item_info_batch=None):
        return self._classifier.classify_batch(text_batch, item_info_batch)

def main():
    model_path = '../data/ssss_haokan/'
    train_path = model_path + 'train_haokan.csv'
    test_path = model_path + 'tengxun_text.csv'

    #cc = ConfigClassifier(train_path, model_path)
    cc = ConfigClassifier(module_dir=model_path)
    #cc.train()
    cc.load_model()
    cc.predict(test_path)
    #cc.evaluation(test_path)

if __name__ == '__main__':
    main()    

        
        
