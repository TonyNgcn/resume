#!/usr/bin/python 
# -*- coding: utf-8 -*-

import threading
import os
import config
from preprocess.bert_sentencepre import preprocess as spreprocess
# from preprocess.wordpre import preprocess as wpreprocess
from model.bert_sentencerec import BertSentenceRecModel
# from model.wordrec import defualt_model as wmodel

def do_deal_bert_sentencerec_model():
    filenames = os.listdir(config.TAG_SR_DIC)
    filepaths = [config.TAG_SR_DIC + "/" + filename for filename in filenames if filename != ".DS_Store"]

    train_x, train_y, test_x, test_y = spreprocess.deal_tagdata(filepaths)
    spreprocess.save_data("strain_x.npy", train_x)
    spreprocess.save_data("strain_y.npy", train_y)
    spreprocess.save_data("stest_x.npy", test_x)
    spreprocess.save_data("stest_y.npy", test_y)


def do_deal_sentencerec_model():
    filenames = os.listdir(config.TAG_SR_DIC)
    filepaths = [config.TAG_SR_DIC + "/" + filename for filename in filenames if filename != ".DS_Store"]

    train_x, train_y, test_x, test_y = spreprocess.deal_tagdata(filepaths)
    spreprocess.save_data("strain_x.npy", train_x)
    spreprocess.save_data("strain_y.npy", train_y)
    spreprocess.save_data("stest_x.npy", test_x)
    spreprocess.save_data("stest_y.npy", test_y)


def deal_sentencerec_model():
    thread = threading.Thread(target=do_deal_sentencerec_model)
    thread.start()
    return thread


def do_deal_wordrec_model():
    filenames = os.listdir(config.TAG_WR_DIC)
    filepaths = [config.TAG_WR_DIC + "/" + filename for filename in filenames if filename != ".DS_Store"]

    train_x, train_y, test_x, test_y = wpreprocess.deal_tagdata(filepaths)
    wpreprocess.save_data("wtrain_x.npy", train_x)
    wpreprocess.save_data("wtrain_y.npy", train_y)
    wpreprocess.save_data("wtest_x.npy", test_x)
    wpreprocess.save_data("wtest_y.npy", test_y)


def deal_wordrec_model():
    thread = threading.Thread(target=do_deal_wordrec_model())
    thread.start()
    return thread


if __name__ == '__main__':
    do_deal_bert_sentencerec_model()

    model = BertSentenceRecModel()
    model.train()
    model.test()

    # wmodel.train()
    # wmodel.test()
