#!/usr/bin/python 
# -*- coding: utf-8 -*-

import logging
import os
import jieba

import config
from tool import bigfile
from holder.jiebah import jieba_holder


class SplitWord(object):
    # 将中文wiki语料库分词，保存到predata目录下
    def wikisplit2word(self):
        logging.info("wiki corpus split to word")
        if os.path.exists(config.CORPUS_DIC + "/wiki_chs"):
            with open(config.PREDATA_DIC + "/totalpart.txt", "w", encoding="utf-8") as write_file:
                for line in bigfile.get_lines(config.CORPUS_DIC + "/wiki_chs"):
                    if line:
                        try:
                            write_file.write(" ".join(jieba.lcut(line)))
                        except Exception as e:
                            if isinstance(e, KeyboardInterrupt):
                                exit(1)
                            logging.warning("error line:{}".format(line))
        else:
            logging.error("file {} is not exist".format(config.CORPUS_DIC + "/wiki_chs"))

    # 其他语料库分词
    def othersplit2word(self, filepath: str):
        logging.info("other corpus split to word")
        if os.path.exists(filepath):
            with open(config.PREDATA_DIC + "/" + filepath.split("/")[-1], "w", encoding="utf-8") as write_file:
                for line in bigfile.get_lines(filepath):
                    if line:
                        try:
                            write_file.write(" ".join(jieba_holder.lcut(line)))
                        except Exception as e:
                            if isinstance(e, KeyboardInterrupt):
                                exit(1)
                            logging.warning("error line:{}".format(line))
        else:
            logging.error("file {} is not exist".format(filepath))


default_splitword = SplitWord()
