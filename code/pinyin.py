#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

from Pinyin2Hanzi import DefaultDagParams
from Pinyin2Hanzi import dag
import pypinyin
from pypinyin import lazy_pinyin


def pinyin_sentence(sentences):
    pinyin = lazy_pinyin(sentences) #,style=pypinyin.TONE3
    return pinyin   #" ".join(pinyin) 



def pinyin_2_hanzi(sentences):
    from Pinyin2Hanzi import DefaultDagParams
    from Pinyin2Hanzi import dag
    dagParams = DefaultDagParams()
    pinyinList = lazy_pinyin(sentences)
    print(pinyinList)
    result = dag(dagParams, pinyinList, path_num=3) #10代表侯选值个数
    for item in result:
        socre = item.score 
        res = item.path # 转换结果
        print(socre, res)
if __name__ == '__main__':
    lists1 = ['jing', 'chang']
    sentences = ["经常","寻衅滋事",'检察机关','提起','忽略']
    for sent in sentences:
        pinyin_2_hanzi(sent)
        print()