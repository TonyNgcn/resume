#!/usr/bin/python 
# -*- coding: utf-8 -*-

import os
import logging
import argparse

import config
from tag.sentencetag import default_tag as stag
from tag.wordtag import default_tag as wtag


# 使用模型来标注特征句模型数据 传入源简历文件夹中的文件名
def tag_sentence(filename: str):
    filepath = os.path.join(config.SRCDATA_DIC, filename)
    logging.info("tag sentence data begin")
    stag.tag(filepath)
    logging.info("tag sentence data finish")


def tag_word(filename: str):
    filepath = os.path.join(config.SRCDATA_DIC, filename)
    logging.info("tag word data begin")
    wtag.tag(filepath)
    logging.info("tag word data finish")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", type=str, help="type of tag between sentence and word")
    parser.add_argument("-f", "--filename", type=str, help="the file name that need to tag")
    args = parser.parse_args()

    tag_type = args.type
    tag_filename = args.filename
    if tag_type == "sentence":
        tag_sentence(tag_filename)
    elif tag_type == "word":
        tag_word(tag_filename)
    else:
        logging.critical("type error")


if __name__ == '__main__':
    main()
