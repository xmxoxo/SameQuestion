#!/usr/bin/env python3
# coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import os
import pandas as pd
import numpy as np
from itertools import combinations
import xmltodict

from pandas.core.frame import DataFrame
from nlpEDA import *


# 读入文件
def readtxtfile(fname, encoding='utf-8'):
    pass
    try:
        with open(fname, 'r', encoding=encoding) as f:
            data = f.read()
        return data
    except:
        return ''


# 保存文本信息到文件
def savetofile(txt, filename, encoding='utf-8'):
    pass
    try:
        with open(filename, 'w', encoding=encoding) as f:
            f.write(str(txt))
        return 1
    except:
        return 0


# 合并测试集数据
def mergeDev():
    path = '../data/'
    dev_dat = os.path.join(path, 'dev_set.csv')
    dev_sub = os.path.join(path, 'sample_submission.csv')
    dev_out = os.path.join(path, 'test.tsv')

    df_dat = pd.read_csv(dev_dat, delimiter="\t")
    df_sub = pd.read_csv(dev_sub, delimiter="\t", header=None)
    df_sub.columns = ['qid', 'label']
    # print(df_sub.head())

    df_out = pd.merge(df_dat, df_sub)
    df_out.drop('qid', axis = 1, inplace = True)
    print(df_out.head())
    df_out.to_csv(dev_out, sep="\t", index=0, header=None)
    print('OK! save to %s' % dev_out)


'''
生成训练数据集

数据格式：
	<TrainCorpus>
		<Questions number="0">
			<EquivalenceQuestions>
				<question>哪些情形下，不予受理民事诉讼申请？</question>
				<question>民事诉讼中对哪些情形的起诉法院不予受理</question>
				<question>人民法院不予受理的民事案件有哪些情形？</question>
			</EquivalenceQuestions>
			<NotEquivalenceQuestions>
				<question>民事诉讼什么情况下不能立案</question>
				<question>哪些案件会给开具民事诉讼不予立案通知书</question>
				<question>法院对于哪些案件再审申请不予受理</question>
			</NotEquivalenceQuestions>
		</Questions>
'''


# 把值两两组合，形成dataframe，添加到DF中，然后返回DF
# 列名： question1	question2	label
def append_DataFrame(lstEqu, lstNotEqu):
    equ = []
    notequ = []
    for i in list(combinations(lstEqu,2)):
        equ.append(list(i))
        # 2019/11/19 增加 “正2 正1 1” 的样本
        equ.append(list(i)[::-1])
    for que in lstEqu:
        for nque in lstNotEqu:
            notequ.append([que,nque])
            #notequ.append([nque,que])
    lstee = np.c_[equ , np.ones((len(equ),1),dtype=np.int)]
    lstnot = np.c_[notequ , np.zeros((len(notequ), 1),dtype=np.int)]
    data = np.r_[lstee,lstnot]
    #print(data)
    #print(data.shape)
    return data


# 数据预处理，生成训练数据并拆分好验证集
def CreateTrain(fold_k = 0):
    path = '../data/'
    train_xml = os.path.join(path, 'train_set.xml')
    out_file = os.path.join(path, 'train_all.tsv')

    txt_xml = readtxtfile(train_xml)
    ojson = xmltodict.parse(txt_xml)
    length_questions = len(ojson['TrainCorpus']['Questions'])
    print('Questions Count: %d' % length_questions)
    print(ojson['TrainCorpus']['Questions'][0]['EquivalenceQuestions']['question'])
    print(ojson['TrainCorpus']['Questions'][0]['NotEquivalenceQuestions']['question'])
    print('-'*30)
    # print(json.dumps(ojson, indent=4))

    # 组合所有匹配
    data_all = []
    for i in range(length_questions):
        lst_Equ = ojson['TrainCorpus']['Questions'][i]['EquivalenceQuestions']['question']
        lst_NotEqu = ojson['TrainCorpus']['Questions'][i]['NotEquivalenceQuestions']['question']
        # 2019/11/15 原始数据中含有空数据，要进行清除。 
        # 测试用例： Questions number="3316"
        lst_Equ = list(filter(None, lst_Equ))
        lst_NotEqu = list(filter(None, lst_NotEqu))
        # 2019/11/20 原始数据中会含有重复的数据，要进行过滤
        lst_Equ = list(set(lst_Equ))
        lst_NotEqu = list(set(lst_NotEqu))

        data = append_DataFrame(lst_Equ, lst_NotEqu)
        data_all.extend(data)

    df_out = DataFrame(data_all, columns=['question1', 'question2', 'lable'])
    print(df_out.head())
    df_out.to_csv(out_file, sep="\t", index=0, header=None)
    print('训练数据总数量：%d' % df_out.shape[0])
    print('-'*30)

    # 根据KFold拆分值分别处理 2019/11/20

    if fold_k == 0:
        # 打乱拆分成train和dev，比例 8:2
        rate = 0.8 
        df_out = df_out.sample(frac=1, random_state=29)
        h = int(rate * df_out.shape[0])
        t = df_out.shape[0] - h
        print('%d==>%d,%d' % (df_out.shape[0], h,t) )
        df_train = df_out.head(h)
        df_dev = df_out.tail(t)

        '''
        # 另一种拆分方法，但是数据会重复
        df_train = df_out.sample(frac=0.8, replace=False, random_state=29)
        df_dev = df_out.sample(frac=0.2, replace=False, random_state=29)
        '''
        # 保存数据 
        out_train = os.path.join(path, 'train.tsv')
        out_dev = os.path.join(path, 'dev.tsv')

        df_train.to_csv(out_train, sep="\t", index=0, header=None)
        df_dev.to_csv(out_dev, sep="\t", index=0, header=None)
    else:
        # 增加K Fold拆分方法 2019/11/20
        from sklearn.model_selection import KFold
        # prepare cross validation
        kfold = KFold(n_splits=fold_k, shuffle=True, random_state=29)
        # enumerate splits
        i = 0
        # create folder
        dat_path = os.path.join(path, 'kfold/')
        if not os.path.exists(dat_path):
            os.mkdir(dat_path)
        for train, dev in kfold.split(df_out):
            i += 1
            print('正在生成K Fold,第%d份数据集...' % i)
            df_train = df_out.iloc[train]
            df_dev = df_out.iloc[dev]
            
            out_path = os.path.join(dat_path, 'model_%d/' % i )
            if not os.path.exists(out_path):
                os.mkdir(out_path)

            # 复制test.tsv文件
            #print('xcopy "%s" "%s"' % ( os.path.abspath( os.path.join(path, 'test.tsv')), os.path.abspath(out_path)))
            os.system('xcopy "%s" "%s"' % ( os.path.abspath( os.path.join(path, 'test.tsv')), os.path.abspath(out_path)))

            # 生成文件名
            out_train = os.path.join(out_path, 'train.tsv')
            out_dev = os.path.join(out_path, 'dev.tsv')

             # 保存数据 
            df_train.to_csv(out_train, sep="\t", index=0, header=None)
            df_dev.to_csv(out_dev, sep="\t", index=0, header=None)

    print('数据生成完毕...')   
   

if __name__ == '__main__':
    pass
    mergeDev()
    CreateTrain(fold_k=5)

