#!/usr/bin/python 
# -*- coding: utf-8 -*-

import os
import logging

import config
from tool import bigfile
from preprocess.bert_wordpre_test import WordRecDataLoader, default_preprocess as wpreprocess
from model.bert_wordrec import BertWordRecModel
from tag.wordtag import CotrainWordRecTag


class CoTrainWordRecDataLoader(object):
    def __init__(self, filename: str):
        self._filename = filename

    def _get_data(self):
        datas = list()
        tagdata_filepath = os.path.join(config.TMP_WR_DIC, self._filename)
        if os.path.exists(tagdata_filepath):
            for line in bigfile.get_lines(tagdata_filepath):
                datas.append(line)
        else:
            logging.warning("tag data file {} is not exist".format(tagdata_filepath))

        return datas

    def _split_data(self, datas: list):
        if len(datas) % 2 == 1:
            logging.error("datas length error")

        chars_list = list()  # 保存分词后的句子 每一项是字符串： 字 字
        labels_list = list()  # 保存标签 每一项是字符串： label label

        # 奇数行是分好词的句子，偶数行是对应的词标签
        single = True  # 判断是奇数行还是偶数行 True 为奇数行
        words = list()

        for line in datas:
            if line:
                if single:
                    words = line.strip("\n").split(" ")
                    # 要判断是否乱行
                    single = False
                else:
                    labels = line.strip("\n").split(" ")
                    single = True

                    if len(words) == len(labels) and len(words) != 0:
                        chars_list.append(words)
                        labels_list.append(labels)
                    else:
                        logging.warning("error data: {} {}".format(words, labels))

        return chars_list, labels_list

    def split_data(self):
        return self._split_data(self._get_data())


class DataLoader(object):
    def __init__(self, embeddings, indexs_list):
        self._embeddings = embeddings
        self._indexs_list = indexs_list

    def get_batch_traindata(self, batch_size):
        total_size = len(self._embeddings)
        start = 0
        while start + batch_size < total_size:
            yield self._embeddings[start:start + batch_size], self._indexs_list[start:start + batch_size]
            start += batch_size
        if len(self._embeddings[start:]) > 0:
            yield self._embeddings[start:], self._indexs_list[start:]
        pass


def cotrain():
    loader = WordRecDataLoader()

    train_chars_list, train_labels_list = loader.get_traindata()
    total_train_len = len(train_chars_list)
    first_train_len = int(total_train_len / 3)
    second_train_len = int(total_train_len * (2 / 3.0))

    # 第一个模型
    first_train_chars_list = train_chars_list[:first_train_len]
    first_train_labels_list = train_labels_list[:first_train_len]

    first_embeddings = wpreprocess.chars2embeddings(first_train_chars_list)
    first_indexs_list = wpreprocess.labels2indexs(first_train_labels_list)

    first_train_chars_list = None
    first_train_labels_list = None

    first_model = BertWordRecModel(model_name="first_bert_wordrec.ckpt")
    first_model.train(train_generator=DataLoader(first_embeddings, first_indexs_list).get_batch_traindata)
    first_embeddings = None
    first_indexs_list = None
    first_model.test()

    # 第二个模型
    second_train_chars_list = train_chars_list[first_train_len:second_train_len]
    second_train_labels_list = train_labels_list[first_train_len:second_train_len]

    second_embeddings = wpreprocess.chars2embeddings(second_train_chars_list)
    second_indexs_list = wpreprocess.labels2indexs(second_train_labels_list)

    second_train_chars_list = None
    second_train_labels_list = None

    second_model = BertWordRecModel(model_name="second_bert_wordrec.ckpt")
    second_model.train(train_generator=DataLoader(second_embeddings, second_indexs_list).get_batch_traindata)
    second_embeddings = None
    second_indexs_list = None
    second_model.test()

    # 第三个模型
    third_train_chars_list = train_chars_list[second_train_len:]
    third_train_labels_list = train_labels_list[second_train_len:]

    third_embeddings = wpreprocess.chars2embeddings(third_train_chars_list)
    third_indexs_list = wpreprocess.labels2indexs(third_train_labels_list)

    third_train_chars_list = None
    third_train_labels_list = None

    third_model = BertWordRecModel(model_name="third_bert_wordrec.ckpt")
    third_model.train(train_generator=DataLoader(third_embeddings, third_indexs_list).get_batch_traindata)
    third_embeddings = None
    third_indexs_list = None
    third_model.test()

    filenames = os.listdir(config.SRCDATA_DIC)
    for filename in filenames:
        if filename not in ["resume_data1.txt", "resume_data2.txt", ".D_Store"]:
            filepath = os.path.join(config.SRCDATA_DIC, filename)
            print(filepath)

            # 预测未标注数据
            first_tag = CotrainWordRecTag(model_name="first_bert_wordrec.ckpt")
            first_tag.tag(filepath, "first")

            second_tag = CotrainWordRecTag(model_name="second_bert_wordrec.ckpt")
            second_tag.tag(filepath, "second")

            third_tag = CotrainWordRecTag(model_name="third_bert_wordrec.ckpt")
            third_tag.tag(filepath, "third")

            # 选择三个模型预测都相同的数据放入测试数据中，只有两个模型预测相同的数据放到另一个模型的训练数据中
            first_loader = CoTrainWordRecDataLoader("first_" + filename)
            second_loader = CoTrainWordRecDataLoader("second_" + filename)
            third_loader = CoTrainWordRecDataLoader("third_" + filename)

            fisrt_chars_list, first_labels_list = first_loader.split_data()
            second_chars_list, second_labels_list = second_loader.split_data()
            third_chars_list, third_labels_list = third_loader.split_data()

            test_chars_list = []
            test_labels_list = []

            first_train_chars_list = []
            first_train_labels_list = []

            second_train_chars_list = []
            second_train_labels_list = []

            third_train_chars_list = []
            third_train_labels_list = []

            for first_chars, first_labels, second_labels, third_labels in zip(fisrt_chars_list, first_labels_list,
                                                                              second_labels_list, third_labels_list):
                str_first = " ".join(first_labels)
                str_second = " ".join(second_labels)
                str_third = " ".join(third_labels)

                # 三个都相同
                if str_first == str_second == str_third:
                    # test_chars_list.append(first_chars)
                    # test_labels_list.append(first_labels)
                    first_train_chars_list.append(first_chars)
                    first_train_labels_list.append(first_chars)
                    second_train_chars_list.append(first_chars)
                    second_train_labels_list.append(first_labels)
                    third_train_chars_list.append(first_chars)
                    third_train_labels_list.append(first_labels)
                # 第一个和第二个相同
                elif str_first == str_second:
                    # 给第三个添加训练数据
                    third_train_chars_list.append(first_chars)
                    third_train_labels_list.append(first_labels)
                # 第一个和第三个相同
                elif str_first == str_third:
                    # 给第二个添加训练数据
                    second_train_chars_list.append(first_chars)
                    second_train_labels_list.append(first_labels)
                # 第二个和第三个相同
                elif str_second == str_third:
                    # 给第一个添加训练数据
                    first_train_chars_list.append(first_chars)
                    first_train_labels_list.append(second_labels)

            # if os.path.exists(config.PREDATA_DIC + "/contrain_test_x.npy") and os.path.exists(
            #         config.PREDATA_DIC + "/contrain_test_y.npy"):
            #
            #
            # if len(test_chars_list) > 0:
            #     # test_file = open(config.TAG_TMP_WR_DIC + "/test.txt", "a")
            #     # for test_chars, test_labels in zip(test_chars_list, test_labels_list):
            #     #     test_file.write(" ".join(test_chars))
            #     #     test_file.write("\n")
            #     #     test_file.write(" ".join(test_labels))
            #     #     test_file.write("\n")
            #     # test_file.close()
            #
            #     test_embeddings = wpreprocess.chars2embeddings(test_chars_list)
            #     test_indexs_list = wpreprocess.labels2indexs(test_labels_list)

            if len(first_train_chars_list) > 0:
                first_embeddings = wpreprocess.chars2embeddings(first_train_chars_list)
                first_indexs_list = wpreprocess.labels2indexs(first_train_labels_list)
                first_model.train(continue_train=True, train_generator=DataLoader(first_embeddings, first_indexs_list).get_batch_traindata)
                first_embeddings = None
                first_indexs_list = None
                first_model.test()

            if len(second_train_chars_list) > 0:
                second_embeddings = wpreprocess.chars2embeddings(second_train_chars_list)
                second_indexs_list = wpreprocess.labels2indexs(second_train_labels_list)

                second_model.train(continue_train=True,
                                   train_generator=DataLoader(second_embeddings, second_indexs_list).get_batch_traindata)
                second_embeddings = None
                second_indexs_list = None
                second_model.test()

            if len(third_train_chars_list) > 0:
                third_embeddings = wpreprocess.chars2embeddings(third_train_chars_list)
                third_indexs_list = wpreprocess.labels2indexs(third_train_labels_list)

                third_model.train(continue_train=True,
                                  train_generator=DataLoader(third_embeddings, third_indexs_list).get_batch_traindata)
                third_embeddings = None
                third_indexs_list = None
                third_model.test()
