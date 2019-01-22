#!/usr/bin/python 
# -*- coding: utf-8 -*-

import re
import os
import logging
import threading

import config
from preprocess.splitword import default_splitword as splitw
from model.wordvec import WordVecModel


def split_corpus():
    thread = threading.Thread(target=splitw.wikisplit2word)
    thread.start()

    filenames = os.listdir(config.SRCDATA_DIC)
    for filename in filenames:
        logging.info("split word from file {}".format(filename))
        splitw.othersplit2word(config.SRCDATA_DIC + "/" + filename)

    thread.join()


def train_wordvec_model():
    wordvec_model = WordVecModel()

    wordvec_model.train()

    filenames = os.listdir(config.PREDATA_DIC)
    filepaths = [config.PREDATA_DIC + "/" + filename for filename in filenames if re.match(r"resume.*\.txt", filename)]
    wordvec_model.train_more(filepaths)

    wordvec_model.train_more([config.CORPUS_DIC + "/name.txt"])


if __name__ == '__main__':
    split_corpus()
    train_wordvec_model()
