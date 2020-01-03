# coding=utf-8

#from classify_driver import *
from config_classifier import ConfigClassifier
import argparse
import os

parser = argparse.ArgumentParser(description='执行分类模型的训练、预测和评估。')

parser.add_argument('project_dir', help='项目文件夹地址，一般位于../data/目录下')
parser.add_argument('-t','--train', help='训练文件名，必须位于项目文件夹下')
parser.add_argument('-k','--k_fold', type=int, nargs='?', const=3, default=False, help='指定k值对训练集进行k-Fold交叉验证, 默认为3')
parser.add_argument('-e','--evaluation', type=bool, nargs='?', const=True, default=False, help='对有标注类别的测试文件进行评估, 需指定测试文件')
parser.add_argument('-p','--predict', help='测试或待预测文件名，必须位于项目文件夹下')

args = parser.parse_args()

module_dir = args.project_dir
if not os.path.exists(module_dir):
    print '项目文件夹%s不存在，请创建!' % module_dir
    exit(1)

cls = ConfigClassifier(module_dir=module_dir)
if args.train:
    train_path = os.path.join(module_dir, args.train)
    if not os.path.exists(module_dir):
        print '找不到训练文件: %s' % train_path
        exit(1)
    
    cls.set_file_path(train_file_path=train_path)
    cls.train(k_fold=args.k_fold)

if args.predict:
    test_path = os.path.join(module_dir, args.predict)
    if not os.path.exists(module_dir):
        print '找不到测试或待预测文件: %s' % test_path
        exit(1)
    cls.load_model()
    cls.predict(test_path)
    
    if args.evaluation:
        cls.evaluation(test_path)
