#!/usr/bin/python 
# -*- coding: utf-8 -*-

import config
from bert.tokenization import FullTokenizer
from model.mybert import MyBertModel

tokenizer = FullTokenizer(vocab_file=config.CORPUS_DIC + "/vocab.txt")
bert_holder = MyBertModel()
