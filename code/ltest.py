# -*- coding: utf-8 -*-
# @Time : 2019/11/11 16:17
# @Author : FRF
# @FileName: demo_combination.py
# @Software: PyCharm

from itertools import combinations

def test_func1(num_list):
    res_list=[]
    res_list.append(list(combinations(num_list, 2)))
    return  res_list

if __name__ =='__main__':
    num_list= ['哪些情形下，不予受理民事诉讼申请？',\
               '民事诉讼中对哪些情形的起诉法院不予受理',\
               '人民法院不予受理的民事案件有哪些情形？']
    combin_li = test_func1(num_list)
    print(combin_li)






















