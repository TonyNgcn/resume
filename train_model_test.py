#!/usr/bin/python 
# -*- coding: utf-8 -*-

import argparse
import logging

from preprocess.bert_sentencepre import default_preprocess as spreprocess
from preprocess.bert_sentencepre import SentenceRecDataLoader
from model.bert_sentencerec import defualt_model as smodel

from preprocess.bert_wordpre_test import default_preprocess as wpreprocess
from preprocess.bert_wordpre_test import WordRecDataLoader
from model.bert_wordrec import default_model as wmodel


def deal_sentencerec_data():
    loader = SentenceRecDataLoader()
    spreprocess.deal_traindata(loader)
    spreprocess.deal_testdata(loader)
    spreprocess.deal_valdata(loader)


def train_sentencerec_model():
    smodel.train()
    smodel.test()


def deal_wordrec_data():
    loader = WordRecDataLoader()
    wpreprocess.deal_traindata(loader)
    wpreprocess.deal_testdata(loader)
    wpreprocess.deal_valdata(loader)


def train_wordrec_model():
    wmodel.train()
    wmodel.test()


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("-t", "--type", type=str,
                       help="sentence is for training sentence recognition model, word is for training word recognition model!")
    args = parse.parse_args()
    if args.type == "sentence":
        deal_sentencerec_data()
        train_sentencerec_model()
    elif args.type == "word":
        deal_wordrec_data()
        train_wordrec_model()
    else:
        logging.critical("type error")


if __name__ == '__main__':
    main()
