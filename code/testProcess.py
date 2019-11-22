#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import os
import pandas as pd
import sys


# 简化版 计算模型目录下的预测结果文件
def testProcess (model):
    path = '../data/model-' + model
    if not os.path.exists(path):
        print('目录%s不存在，请检查!' % path)
        return
    CalcProcess (path)

# 计算指定目录下的预测文件
def CalcProcess (path):
    test_dat = os.path.join(path, 'test_results.tsv')
    if not os.path.exists(test_dat):
        print('文件%s不存在，请检查!' % test_dat)
        return

    out_test = os.path.join(path, 'result.csv')
   
    df_test = pd.read_csv(test_dat, delimiter="\t", header=None)
    df_test['result'] = df_test[0].apply(lambda x:int(x<0.5))
    print(df_test.head(10))
    df_test = df_test[['result']]
    df_test.to_csv(out_test, sep="\t", header=0) #,encoding='utf_8'     utf_8_sig
    print('数据已生成:%s' % out_test)
    

if __name__ == '__main__':
    pass
    num = ''
    if len(sys.argv)>1:
        num = sys.argv[1]
    if not num:
        print('请指定模型序号！\n 命令行用法：testProcess.py 01')
    else:
        testProcess(num)

