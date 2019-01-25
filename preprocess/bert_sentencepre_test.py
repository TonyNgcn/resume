#!/usr/bin/python 
# -*- coding: utf-8 -*-

import logging
import numpy as np
import os

import config
from tool import shuffle, bigfile
from holder.berth import tokenizer, bert_holder


# 用来加载数据
class SentenceRecDataLoader(object):
    # get_traindata()和get_testdata()函数一定要有
    def __init__(self, rate: float = config.SR_RATE):
        if rate <= 0 or rate >= 1:
            logging.critical("rate must between 0 and 1")
        self._rate = rate

    # 获取数据
    def _get_data(self):
        datas = list()
        tagdata_filenames = os.listdir(config.TAG_SR_DIC)
        tagdata_filepaths = [config.TAG_SR_DIC + "/" + tagdata_filename
                             for tagdata_filename in tagdata_filenames if tagdata_filename != ".D_Store"]
        for tagdata_filepath in tagdata_filepaths:
            if os.path.exists(tagdata_filepath):
                for line in bigfile.get_lines(tagdata_filepath):
                    datas.append(line)
            else:
                logging.warning("tag data file {} is not exist".format(tagdata_filepath))
        return datas

    # 切分数据
    def _split_data(self, datas: list):
        sentences = list()  # 保存分词后的句子
        labels = list()  # 保存标签

        for line in datas:
            if line:
                pair = line.strip("\n").split(";;;")  # 将句子和标签分开
                if len(pair) == 2 and pair[0] and pair[1]:
                    logging.debug("句子:{} 标签:{}".format(pair[0], pair[1]))
                    sentences.append(pair[0])
                    labels.append(pair[1])
                else:
                    logging.warning("error line {}".format(line))
            else:
                logging.warning("error line {}".format(line))
        return sentences, labels  # 返回 句子列表，表示该句句子的标签列表

    # 获取训练数据
    def get_traindata(self):
        datas = self._get_data()
        datas_len = len(datas)
        traindata_len = int(datas_len * self._rate)
        traindatas = datas[:traindata_len]
        return self._split_data(traindatas)

    # 测试训练数据
    def get_testdata(self):
        datas = self._get_data()
        datas_len = len(datas)
        traindata_len = int(datas_len * self._rate)
        traindatas = datas[traindata_len:]
        return self._split_data(traindatas)


# 用来处理数据成输入
class SentenceRecPreprocess(object):
    _total_labels = ['ctime', 'ptime', 'basic', 'wexp', 'sexp', 'noinfo']  # 标签列表

    # 修正数据
    def fix(self, sentences: list, labels: list):
        def label_in(label: str):
            if label in self._total_labels:
                return True
            return False

        fix_sentences = []
        fix_labels = []
        for sentence, label in zip(sentences, labels):
            if label_in(label):
                fix_sentences.append(sentence)
                fix_labels.append(label)
            else:
                logging.warning("label {} not in total labels".format(label))

        return fix_sentences, fix_labels

    ###########################################################################
    # 将句子转成字
    def _split2char(self, sentences: list):
        chars_list = []
        for sentence in sentences:
            chars = tokenizer.tokenize(sentence)
            chars_list.append(chars)
        return chars_list

    # 截取字
    def _regchar(self, chars_list: list):
        regchars_list = []
        for chars in chars_list:
            regchars = []
            chars_len = len(chars)
            regchars.append("[CLS]")
            if chars_len >= config.SENTENCE_LEN:
                regchars.extend(chars[:config.SENTENCE_LEN])
            else:
                regchars.extend(chars)
                for _ in range(config.SENTENCE_LEN - chars_len):
                    regchars.append("[PAD]")
            regchars.append("[SEP]")
            regchars_list.append(regchars)
        return regchars_list

    # 转成input_ids
    def _to_input_ids(self, chars_list: list):
        input_ids_list = []
        for chars in chars_list:
            ids = tokenizer.convert_tokens_to_ids(chars)
            input_ids_list.append(ids)
        return input_ids_list

    # 转成embeddings
    def _to_embeddings(self, input_ids: list):
        embeddings = bert_holder.predict(input_ids)
        return embeddings

    # 句子转embeddings
    def sentences2embeddings(self, sentences: list):
        logging.info("sentences to embeddings")
        chars_list = self._split2char(sentences)
        regchars_list = self._regchar(chars_list)
        input_ids_list = self._to_input_ids(regchars_list)
        embeddings = self._to_embeddings(input_ids_list)
        return embeddings

    ###########################################################################
    # 单个标签转向量
    def _to_vector(self, label: str):
        vector = list()

        if label in self._total_labels:
            for l in self._total_labels:
                if l == label:
                    vector.append(1.0)
                else:
                    vector.append(0.0)
        else:
            # 标签不存在就设为noinfo的one hot向量
            logging.warning("label {} is not exist".format(label))
            for _ in range(len(self._total_labels) - 1):
                vector.append(0.0)
            vector.append(1.0)
        return vector

    # 转成one-hot向量
    def labels2vectors(self, labels: list):
        logging.info("labels to one hot vectors")
        vectors = []
        for label in labels:
            vector = self._to_vector(label)
            vectors.append(vector)
        return vectors

    ###########################################################################
    # 获取标签列表
    def get_total_labels(self):
        return self._total_labels

    # 获取标签
    def get_label(self, labelvec):
        maxindex = np.argmax(labelvec)

        try:
            return self._total_labels[int(maxindex)]
        except:
            logging.warning("label vector {} error".format(labelvec))
            return self._total_labels[-1]

    ###########################################################################
    # 加载训练数据
    def _load_traindata(self):
        try:
            strain_x = np.load(config.PREDATA_DIC + '/bert_strain_x.npy')
            strain_y = np.load(config.PREDATA_DIC + '/bert_strain_y.npy')
            return strain_x, strain_y
        except Exception as e:
            logging.error(e)
            exit(1)

    # 加载测试数据
    def _load_testdata(self):
        try:
            stest_x = np.load(config.PREDATA_DIC + '/bert_stest_x.npy')
            stest_y = np.load(config.PREDATA_DIC + '/bert_stest_y.npy')
            return stest_x, stest_y
        except Exception as e:
            logging.error(e)
            exit(1)

    # 获取打乱后的训练数据
    def get_traindata(self):
        strain_x, strain_y = self._load_traindata()

        strain_x, strain_y = shuffle.shuffle_both(strain_x, strain_y)  # 打乱数据

        if len(strain_x) > 0:
            return strain_x, strain_y
        else:
            logging.error("train data length is less than 0")
            exit(1)

    # 获取打乱后的测试数据
    def get_testdata(self):
        stest_x, stest_y = self._load_testdata()

        stest_x, stest_y = shuffle.shuffle_both(stest_x, stest_y)  # 打乱数据

        if len(stest_x) > 0:
            return stest_x, stest_y
        else:
            logging.error("test data length is less than 0")
            exit(1)

    # 批量获取打乱后的训练数据
    def get_batch_traindata(self, batch_size: int):
        strain_x, strain_y = self.get_traindata()

        total_size = len(strain_x)
        start = 0
        while start + batch_size < total_size:
            yield strain_x[start:start + batch_size], strain_y[start:start + batch_size]
            start += batch_size
        if len(strain_x[start:]) > 0:
            yield strain_x[start:], strain_y[start:]

    # 批量获取打乱后的测试数据
    def get_batch_testdata(self, batch_size: int):
        stest_x, stest_y = self.get_testdata()

        total_size = len(stest_x)
        start = 0
        while start + batch_size < total_size:
            yield stest_x[start:start + batch_size], stest_y[start:start + batch_size]
            start += batch_size
        if len(stest_x[start:]) > 0:
            yield stest_x[start:], stest_y[start:]

    def _save_data(self, filename: str, data: list):
        try:
            if len(data) == 0:
                logging.warning("data length is 0")
                return
            np.save(config.PREDATA_DIC + "/" + filename, np.array(data))
            logging.info("save data file {} sucess".format(filename))
        except Exception as e:
            logging.error(e)
            exit(1)

    # 删除训练数据
    def remove_traindata(self):
        try:
            os.remove(config.PREDATA_DIC + "/bert_strain_x.npy")
            os.remove(config.PREDATA_DIC + "/bert_strain_y.npy")
            logging.info("remove train data success")
        except Exception as e:
            logging.warning(e)

    # 删除测试数据
    def remove_testdata(self):
        try:
            os.remove(config.PREDATA_DIC + "/bert_stest_x.npy")
            os.remove(config.PREDATA_DIC + "/bert_stest_y.npy")
            logging.info("remove test data success")
        except Exception as e:
            logging.warning(e)

    ###########################################################################
    # 处理标注的训练数据
    def deal_traindata(self, loader):
        sentences, labels = loader.get_traindata()
        fix_sentences, fix_labels = self.fix(sentences, labels)
        embeddings = self.sentences2embeddings(fix_sentences)
        vectors = self.labels2vectors(fix_labels)
        self._save_data("bert_strain_x.npy", embeddings)
        self._save_data("bert_strain_y.npy", vectors)

    # 处理标注的测试数据
    def deal_testdata(self, loader):
        sentences, labels = loader.get_testdata()
        fix_sentences, fix_labels = self.fix(sentences, labels)
        embeddings = self.sentences2embeddings(fix_sentences)
        vectors = self.labels2vectors(fix_labels)
        self._save_data("bert_stest_x.npy", embeddings)
        self._save_data("bert_stest_y.npy", vectors)


default_preprocess = SentenceRecPreprocess()

if __name__ == '__main__':
    loader = SentenceRecDataLoader()
    preprocess = SentenceRecPreprocess()
    preprocess.deal_traindata(loader)
    preprocess.deal_testdata(loader)
