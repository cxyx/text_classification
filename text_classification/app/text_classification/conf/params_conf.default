#!/usr/bin/env python 
# coding:utf-8

'''
根据配置进行特征选择和模型训练
兼顾规则分类器

example:
	params_dict = {
	'feature_method': [
                    'ngram',
                    'wordseg',
                    'skipgram',
                    'embedding',
                    'dict'
                ],
    'feature_args'  : [
				{   # ngram
					'n_range':(1,3),
					'min_df':3
				},
				{   # wordseg
					'n_range':(1,2),
					'min_df':3,
                    'normalize':False
				},
				{   # skipgram
					'n_range':(1,5),
					'min_df':2,
					'skip_n':1
				},
                {   # embedding
                    'embedding_dim':128,
                    'workers':8,
                    'iter':10
                },
			},
	'classifier'   : [
                    'lr',
                    'lr_rule',
                    'svm',
                    'svm_rule',
                    'multi',
                    'multi_rule',
            ],
	'classifier_args': {
				{   # svm / svm_rule
					'kernal':'linear',
					'C':10.0,
                    ...
				},
				{   # lr / lr_rule / multi / multi_rule
                    'solver':'lbfgs',
					'C':10.0,
                    ...
				},
			}
}
'''

params_dict = {
	'feature_method':['ngram'],
	'feature_args'	:[{'n_range':(1,2),'min_df':5,}],
	'classifier'	:'lr_rule',
	'classifier_args':{'C':15.0}
}
