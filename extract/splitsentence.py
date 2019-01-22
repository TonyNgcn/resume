#!/usr/bin/python 
# -*- coding: utf-8 -*-

import re


# 将单条简历分句
def resume2sentences(srcresume: str):
    # 停用掉某些符号 《》<>（）()「」{}|

    pattern = r'[(](.*?)[)]|[（](.*?)[）]|[《》<>「」{}【】\[\]"“”]'  # 将'《》<>「」{}'"“” 去掉 （）及其里面内容去掉去掉
    pat = re.compile(pattern)
    srcresume = re.sub(pat, '', srcresume)

    # 分句 ，。？！；
    pattern1 = r'[\|\|:：,，.。?？!！;；\n\t\r]'
    pat1 = re.compile(pattern1)
    sentences = re.split(pat1, srcresume.strip())  # 以'，。？！；'为句子分隔符分割句子

    return [sentence for sentence in sentences if sentence]
