#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

'''
EDA: NLP数据增强：
1. 同义词替换（SR: Synonyms Replace）：不考虑stopwords，在句子中随机抽取n个词，然后从同义词词典中随机抽取同义词，并进行替换。
2. 随机插入(RI: Randomly Insert)：不考虑stopwords，随机抽取一个词，然后在该词的同义词集合中随机选择一个，插入原句子中的随机位置。该过程可以重复n次。
3. 随机交换(RS: Randomly Swap)：句子中，随机选择两个词，位置交换。该过程可以重复n次。
4. 随机删除(RD: Randomly Delete)：句子中的每个词，以概率p随机删除。
'''

import random
import jieba
from random import shuffle


class NlpEda:
    def __init__(self, swap_num=2, drop_num=1):
        pass
        self.swap_num = swap_num
        self.drop_num = drop_num

    # lst_segment：分词后的List
    def rand_swap(self, lst_segment):
        pass
        # 随机交换,生成随机交换的索引号对；
        l_index = [i for i in range(len(lst_segment)) ]
        rand_index = []
        for i in range(self.swap_num):
            shuffle(l_index)
            rand_index.append(l_index[:2])
            l_index.pop(0)
            l_index.pop(0)

        # print(rand_index)
        for index in rand_index:
            lst_segment[index[0]], lst_segment[index[1]] = lst_segment[index[1]], lst_segment[index[0]]
        return lst_segment

    # lst_segment 分词后的List
    def rand_delete(self, lst_segment):
        pass
        # 随机删除词
        num_word = len(lst_segment)
        for i in range(self.drop_num):
            rand_index = random.randint(0, num_word-1)
            lst_segment.pop(rand_index)
        return lst_segment

    # 统一方法
    def eda_process(self, segment, num=1):
        lst_dat = list(jieba.cut(segment))
        ret_dat = []
        for i in range(num):
            lst_temp = lst_dat.copy()
            lst_temp = self.rand_swap(lst_temp)
            lst_temp = self.rand_delete(lst_temp)
            ret_dat.append(''.join(lst_temp))

        return ret_dat

    # 数据增强，处理数组,传递数组，返回数组
    def eda_list(self, seg_list, num=1):
        pass
        ret_list = []
        for seg in seg_list:
            ret_list.append(seg)
            new_seg = self.eda_process(seg, num)
            ret_list.extend(new_seg)
        return ret_list


def test():
    eda = NlpEda()
    seg_list = ['租赁合同应当以什么形式订立',
                '我被人家车撞了，但是他的车也是借的别人的，那我是不是可以去找原车主索要赔偿呢',
                '哪些情形下，不予受理民事诉讼申请？',
                '交通事故未划分责任允许放车吗',
                ]
    print('NLP EDA Demo:')
    for seg in seg_list:
        new_seg = eda.eda_process(seg, 4)
        print('%s\n%s' % (seg, '-' * 40))
        for n_seg in new_seg:
            print(n_seg)
        print()

    print('='*30)
    n_list = eda.eda_list(seg_list, 4)
    print(n_list)


if __name__ == '__main__':
    pass
    test()


