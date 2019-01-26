#!/usr/bin/python 
# -*- coding: utf-8 -*-

from preprocess.bert_sentencepre import default_preprocess, SentenceRecDataLoader
from model.bert_sentencerec import defualt_model


def deal_sentencerec_data():
    loader = SentenceRecDataLoader()
    default_preprocess.deal_traindata(loader)
    default_preprocess.deal_testdata(loader)


def train_sentencerec_model():
    defualt_model.train()
    defualt_model.test()


if __name__ == '__main__':
    deal_sentencerec_data()
    train_sentencerec_model()
