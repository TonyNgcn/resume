#!/usr/bin/python 
# -*- coding: utf-8 -*-

import logging
import numpy as np
import os

import config
from tool import bigfile, shuffle
from holder.berth import bert_holder, tokenizer


class SentencePreprocess(object):
    _total_labels = ['ctime', 'ptime', 'basic', 'wexp', 'sexp', 'noinfo']  # 标签列表

    # 获取标签列表
    def get_total_labels(self):
        return self._total_labels

    # 获取标签向量
    def get_labelvec(self, label: str):
        vector = list()

        if label in self._total_labels:
            for l in self._total_labels:
                if l == label:
                    vector.append(1.0)
                else:
                    vector.append(0.0)
            return vector
        else:
            logging.error("label {} is not exist".format(label))
            exit(1)

    # 获取标签
    def get_label(self, labelvec):
        maxindex = np.argmax(labelvec)

        try:
            return self._total_labels[int(maxindex)]
        except:
            logging.warning("label vector {} error".format(labelvec))
            return self._total_labels[-1]

    # 加载训练数据
    def load_traindata(self):
        try:
            strain_x = np.load(config.PREDATA_DIC + '/strain_x.npy')
            strain_y = np.load(config.PREDATA_DIC + '/strain_y.npy')
            return strain_x, strain_y
        except Exception as e:
            logging.error(e)
            exit(1)

    # 加载测试数据
    def load_testdata(self):
        try:
            stest_x = np.load(config.PREDATA_DIC + '/stest_x.npy')
            stest_y = np.load(config.PREDATA_DIC + '/stest_y.npy')
            return stest_x, stest_y
        except Exception as e:
            logging.error(e)
            exit(1)

    # 获取打乱后的训练数据
    def get_traindata(self):
        strain_x, strain_y = self.load_traindata()

        strain_x, strain_y = shuffle.shuffle_both(strain_x, strain_y)  # 打乱数据

        if len(strain_x) > 0:
            return strain_x, strain_y
        else:
            logging.error("train data length is less than 0")
            exit(1)

    # 获取打乱后的测试数据
    def get_testdata(self):
        stest_x, stest_y = self.load_testdata()

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

    def save_data(self, filename: str, data: list):
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
            os.remove(config.PREDATA_DIC + "/strain_x.npy")
            os.remove(config.PREDATA_DIC + "/strain_y.npy")
            logging.info("remove train data success")
        except Exception as e:
            logging.warning(e)

    # 删除测试数据
    def remove_testdata(self):
        try:
            os.remove(config.PREDATA_DIC + "/stest_x.npy")
            os.remove(config.PREDATA_DIC + "/stest_y.npy")
            logging.info("remove test data success")
        except Exception as e:
            logging.warning(e)

    # 句子分字
    def sentence2tokens(self, sentences: list):
        tokens_list = []
        for sentence in sentences:
            tokens = []
            tokens.append("[CLS]")
            token = tokenizer.tokenize(sentence)
            tokens.extend(token)
            tokens.append("[SEP]")
            tokens_list.append(tokens)
        return tokens_list

    # 句子分字 返回填充后的tokens列表
    def sentence2regtokens(self, sentences: list):
        regtokens_list = []
        for sentence in sentences:
            tokens = []
            tokens.append("[CLS]")
            token = tokenizer.tokenize(sentence)
            token_len = len(token)
            if token_len >= config.SENTENCE_LEN:
                tokens.extend(token[:config.SENTENCE_LEN])
            else:
                tokens.extend(token)
                for _ in range(config.SENTENCE_LEN - token_len):
                    tokens.append("[PAD]")
            tokens.append("[SEP]")
            logging.debug("tokens:{}".format(tokens))
            regtokens_list.append(tokens)
        return regtokens_list

    # 将tokens转为input_ids
    def tokens2input_ids(self, tokens_list):
        input_ids_list = []
        for tokens in tokens_list:
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            logging.debug("input_ids:{}".format(input_ids))
            input_ids_list.append(input_ids)
        return input_ids_list

    # input_ids转为embedding
    def input_ids2vec(self, input_ids_list: list):
        logging.info("input_ids to vector")
        embeddings = []
        for input_ids in input_ids_list:
            embedding = bert_holder.predict([input_ids])
            logging.debug("embedding:{}".format(embedding))
            embeddings.append(embedding[0])
        return embeddings

    # 标签转向量
    def label2vec(self, labels: list):
        logging.info("label to vector")
        labelvec_list = list()  # 标签向量列表

        for label in labels:
            labelvec = self.get_labelvec(label)
            labelvec_list.append(labelvec)
        return labelvec_list

    # 处理标注数据
    def deal_tagdata(self, tagdata_filepaths: list, rate: float = config.SR_RATE):
        logging.info("begin deal sentence tag data")
        if rate < 0 or rate > 1:
            logging.error("rate is not between 0 and 1")
            exit(1)

        datas = list()
        for tagdata_filepath in tagdata_filepaths:
            if os.path.exists(tagdata_filepath):
                for line in bigfile.get_lines(tagdata_filepath):
                    datas.append(line)
            else:
                logging.warning("tag data file {} is not exist".format(tagdata_filepath))

        # random.shuffle(datas)
        sentences, labels = self._split_tagdata(datas)
        datas = None

        regtokens_list = self.sentence2regtokens(sentences)
        sentences = None

        input_ids = self.tokens2input_ids(regtokens_list)
        regtokens_list = None

        embeddings = self.input_ids2vec(input_ids)
        labelvecs = self.label2vec(labels)
        input_ids = None
        labels = None

        # 将数据保存下来
        total_size = len(embeddings)

        train_x = embeddings[:int(total_size * rate)]
        train_y = labelvecs[:int(total_size * rate)]
        test_x = embeddings[int(total_size * rate):]
        test_y = labelvecs[int(total_size * rate):]
        embeddings = None
        labelvecs = None

        logging.info("deal sentence tag data end")
        return train_x, train_y, test_x, test_y

    def _split_tagdata(self, datas: list):
        sentences = list()  # 保存分词后的句子
        label_list = list()  # 保存标签

        for line in datas:
            if line:
                pair = line.strip("\n").split(";;;")  # 将句子和标签分开
                if len(pair) == 2 and pair[0] and pair[1]:
                    if pair[1] in self._total_labels:
                        # sentence_words = jieba_holder.lcut(pair[0])
                        logging.debug("句子:{} 标签:{}".format(pair[0], pair[1]))
                        sentences.append(pair[0])
                        label_list.append(pair[1])
                    else:
                        logging.warning("error line {}".format(line))
                else:
                    logging.warning("error line {}".format(line))
            else:
                logging.warning("error line {}".format(line))
        return sentences, label_list  # 返回 句子列表，表示该句句子的标签列表


preprocess = SentencePreprocess()
