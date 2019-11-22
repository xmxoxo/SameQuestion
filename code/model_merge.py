# -*- coding: utf-8 -*-
# @Time : 2019/11/21 17:03
# @Author : FRF, xmxoxo
# @FileName: model_integration.py
# @Software: PyCharm


'''
K-fold 最终预测结果合并计算（加权平均）
模型融合计算：
TR = (∑(Ti*ACCi) )/ (∑ACCi)
'''

import  os
import pandas as pd
import numpy as np
from functools import  reduce
#from pandas.core.frame import DataFrame

from testProcess import *


# 读入文件
def readtxtfile(fname,encoding='utf-8'):
    pass
    try:
        with open(fname,'r',encoding=encoding) as f:  
            data=f.read()
        return data
    except :
        return ''

# 模型融合计算
def model_merge (model_path):
    print('正在自动融合模型结果...')
    # 遍历子目录, 得到每个子目录名称
    lstFloders = []
    for dirname in os.listdir(model_path) :
        file_path = os.path.join(model_path, dirname)
        if os.path.isdir( file_path ):
            lstFloders.append (file_path)      

    print('子目录清单:\n%s' % '\n'.join(lstFloders) )
    
    # 读出每个模型的 ACC[i]与预测结果T[i]
    T = []
    acc = []
    for folder in lstFloders:
        eval_file = os.path.join(folder,'eval_results.txt')
        test_file = os.path.join(folder,'test_results.tsv')
        # 必须两个文件同时存在才可以计算，否则跳过
        if os.path.exists(eval_file) and os.path.exists(test_file):
            # 读出ACC
            eval_dat = readtxtfile(eval_file, encoding='GB2312')
            acc_tmp = float(eval_dat.split('\n')[0].split('=')[1].strip())
            acc.append(acc_tmp)

            df_test = pd.read_csv(test_file,sep='\t',header=None)
            T.append(df_test.values.tolist())
    print('-'*30)
    print('ACC: %s' % acc)
    print('T的大小:%d, T[0]的大小：%d '% (len(T),len(T[0])) )
    
    # 加权计算
    Tm = map(lambda x,y: np.array(x)*y, T,acc)
    Tr = reduce(lambda x,y: x+y, Tm)
    TR = list(Tr/sum(acc))

    # 转成 DataFrame 并保存
    df_tr = pd.DataFrame(TR)
    print('计算后的shape：%s' % str(df_tr.shape) )
    tr_file = os.path.join(model_path, 'test_results.tsv')
    df_tr.to_csv(tr_file, sep="\t", index=0, header=0)

    # 调用方法 计算最终提交结果
    print('-'*30)
    CalcProcess(model_path)


if __name__ == '__main__':
    pass
    # model_path = '../data/model-07/'
    model_path = ''
    if len(sys.argv)>1:
        model_path = sys.argv[1]
    if not model_path:
        print('请指定模型目录名！命令行用法：model_merge.py ../data/model-07/')
    else:
        model_merge (model_path)

