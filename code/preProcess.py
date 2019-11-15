#!/usr/bin/env python3
# coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import os
import sys
import re
import json
import random
import time
import string
import pandas as pd
import numpy as np
import traceback
import logging
import shutil
from itertools import combinations
from itertools import product
import xmltodict

from pandas.core.frame import DataFrame


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


# 合并验证集数据
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
    df_out.to_csv(dev_out, sep="\t", index=0)
    print('OK! save to %s' % dev_out)


'''
#生成训练数据集

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
    for que in lstEqu:
        for nque in lstNotEqu:
            notequ.append([que,nque])
    lstee = np.c_[equ , np.ones((len(equ),1),dtype=np.int)]
    lstnot = np.c_[notequ , np.zeros((len(notequ), 1),dtype=np.int)]
    data = np.r_[lstee,lstnot]
    #print(data)
    #print(data.shape)
    return data


def CreateTrain():
    path = '../data/'
    train_xml = os.path.join(path, 'train_set.xml')
    out_file = os.path.join(path, 'train_all.tsv')
    out_train = os.path.join(path, 'train.tsv')
    out_dev = os.path.join(path, 'dev.tsv')

    txtxml = readtxtfile(train_xml)
    ojson = xmltodict.parse(txtxml)
    length_questions = len(ojson['TrainCorpus']['Questions'])
    print('Questions Count: %d' % length_questions)
    print(ojson['TrainCorpus']['Questions'][0]['EquivalenceQuestions']['question'])
    print(ojson['TrainCorpus']['Questions'][0]['NotEquivalenceQuestions']['question'])
    print('-'*30)
    # print(json.dumps(ojson, indent=4))

    # 组合所有匹配
    data_all = []
    for i in range(length_questions):
        lstEqu = ojson['TrainCorpus']['Questions'][i]['EquivalenceQuestions']['question']
        lstNotEqu = ojson['TrainCorpus']['Questions'][i]['NotEquivalenceQuestions']['question']
        data = append_DataFrame(lstEqu, lstNotEqu)
        data_all.extend(data)

    df_out = DataFrame(data_all,columns=['question1','question2','lable'])
    print(df_out.head())
    df_out.to_csv(out_file, sep="\t", index=0)

    #打乱拆分成train和dev，比例 8:2

    #df_out = df_out.sample(frac=1)
    #h = int(0.8* df_out.shape[0])
    #t = df_out.shape[0] - h
    #print(df_out.shape[0], h,t)
    #df_train = df_out.head(h)
    #df_dev = df_out.tail(t)

    df_train = df_out.sample(frac=0.8,replace=False,random_state=29)
    df_dev = df_out.sample(frac=0.2,replace=False,random_state=29)

    df_train.to_csv(out_train, sep="\t", index=0)
    df_dev.to_csv(out_dev, sep="\t", index=0)
   

if __name__ == '__main__':
    pass
    mergeDev()
    CreateTrain()

