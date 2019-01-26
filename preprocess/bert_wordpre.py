#!/usr/bin/python 
# -*- coding: utf-8 -*-

import os
import numpy as np
import logging

import config
from tool import shuffle, bigfile
from holder.berth import tokenizer, bert_holder


class WordPreprocess(object):
    _total_national_labels = [
        "name", "sex", "nationality", "nation", "time", "school", "pro", "degree", "edu", "company", "department",
        "job", "O"]

    _total_labels = ['B-name', 'I-name', 'E-name',
                     'B-sex', 'I-sex', 'E-sex',
                     'B-nationality', 'I-nationality', 'E-nationality',
                     'B-nation', 'I-nation', 'E-nation',
                     'B-time', 'I-time', 'E-time',
                     'B-school', 'I-school', 'E-school',
                     'B-pro', 'I-pro', 'E-pro',
                     'B-degree', 'I-degree', 'E-degree',
                     'B-edu', 'I-edu', 'E-edu',
                     'B-company', 'I-company', 'E-company',
                     'B-department', 'I-department', 'E-department',
                     'B-job', 'I-job', 'E-job',
                     'O']  # 标签列表

    # 获取标签列表
    def get_total_labels(self):
        return self._total_labels

    # 获取标签向量
    def get_labelindex(self, label: str):
        if label in self._total_labels:
            return self._total_labels.index(label)
        else:
            logging.warning("label {} is not exist".format(label))
            return len(self._total_labels) - 1

    # 获取标签
    def get_label(self, index):
        labels = self._total_labels

        try:
            return labels[index]
        except Exception as e:
            logging.error(e)
            exit(1)

    # 加载训练数据
    def load_traindata(self):
        try:
            wtrain_x = np.load(config.PREDATA_DIC + '/bert_wtrain_x.npy')
            wtrain_y = np.load(config.PREDATA_DIC + '/bert_wtrain_y.npy')
            return wtrain_x, wtrain_y
        except Exception as e:
            logging.error(e)
            exit(1)

    # 加载测试数据
    def load_testdata(self):
        try:
            wtest_x = np.load(config.PREDATA_DIC + '/bert_wtest_x.npy')
            wtest_y = np.load(config.PREDATA_DIC + '/bert_wtest_y.npy')
            return wtest_x, wtest_y
        except Exception as e:
            logging.error(e)
            exit(1)

    # 获取打乱后的训练数据
    def get_traindata(self):
        wtrain_x, wtrain_y = self.load_traindata()

        wtrain_x, wtrain_y = shuffle.shuffle_both(wtrain_x, wtrain_y)  # 打乱数据

        if len(wtrain_x) > 0:
            return wtrain_x, wtrain_y
        else:
            logging.error("train data length is less than 0")
            exit(1)

    # 获取打乱后的测试数据
    def get_testdata(self):
        wtest_x, wtest_y = self.load_testdata()

        wtest_x, wtest_y = shuffle.shuffle_both(wtest_x, wtest_y)  # 打乱数据

        if len(wtest_x) > 0:
            return wtest_x, wtest_y
        else:
            logging.error("test data length is less than 0")
            exit(1)

    # 批量获取打乱后的训练数据
    def get_batch_traindata(self, batch_size: int):
        wtrain_x, wtrain_y = self.get_traindata()

        total_size = len(wtrain_x)
        start = 0
        while start + batch_size < total_size:
            yield wtrain_x[start:start + batch_size], wtrain_y[start:start + batch_size]
            start += batch_size
        if len(wtrain_x[start:]) > 0:
            yield wtrain_x[start:], wtrain_y[start:]

    # 批量获取打乱后的测试数据
    def get_batch_testdata(self, batch_size: int):
        wtest_x, wtest_y = self.get_testdata()

        total_size = len(wtest_x)
        start = 0
        while start + batch_size < total_size:
            yield wtest_x[start:start + batch_size], wtest_y[start:start + batch_size]
            start += batch_size
        if len(wtest_x[start:]) > 0:
            yield wtest_x[start:], wtest_y[start:]

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
            os.remove(config.PREDATA_DIC + "/bert_wtrain_x.npy")
            os.remove(config.PREDATA_DIC + "/bert_wtrain_y.npy")
            logging.info("remove train data success")
        except Exception as e:
            logging.warning(e)

    # 删除测试数据
    def remove_testdata(self):
        try:
            os.remove(config.PREDATA_DIC + "/bert_wtest_x.npy")
            os.remove(config.PREDATA_DIC + "/bert_wtest_y.npy")
            logging.info("remove test data success")
        except Exception as e:
            logging.warning(e)

    # 切分标注数据
    def _split_tagdata(self, datas: list):
        if len(datas) % 2 == 1:
            logging.error("datas lenght error")

        total_labels = self._total_national_labels

        def wrong_words(words):
            for word in words:
                if word in total_labels:
                    return True
            return False

        def wrong_labels(labels):
            for label in labels:
                if label not in total_labels:
                    return True
            return False

        words_list = list()  # 保存分词后的句子 每一项是字符串： 词 词
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

                    if len(words) == len(labels) and len(words) != 0 and not wrong_words(words) and not wrong_labels(
                            labels):
                        words_list.append(words)
                        labels_list.append(labels)
                    else:
                        logging.warning("error data: {} {}".format(words, labels))

        return words_list, labels_list

    # 将实体词分字 实体标签也分
    def deal_wls(self, national_words_list: list, national_labels_list: list):
        chars_list = []
        labels_list = []
        for national_words, national_labels in zip(national_words_list, national_labels_list):
            chars, labels = self.deal_wl(national_words, national_labels)
            chars_list.append(chars)
            labels_list.append(labels)
        return chars_list, labels_list

    # 将实体词分字 实体标签也分
    def deal_wl(self, national_words: list, national_labels: list):
        chars = []
        labels = []
        for national_word, national_label in zip(national_words, national_labels):
            w = list(national_word)
            w_len = len(w)
            chars.extend(w)
            if national_label == "O":
                for _ in range(w_len):
                    labels.append("O")
            else:
                if w_len == 1:
                    labels.append("E-" + national_label)
                elif w_len == 2:
                    labels.append("B-" + national_label)
                    labels.append("E-" + national_label)
                elif w_len >= 3:
                    labels.append("B-" + national_label)
                    for _ in range(w_len - 2):
                        labels.append("I-" + national_label)
                    labels.append("E-" + national_label)
                else:
                    logging.error("national_label:{} error".format(national_label))
                    exit(1)
        return chars, labels

    # 字序列转token序列
    def words2regtokens(self, words_list: list):
        regtokens_list = []
        for words in words_list:
            regtokens_list.append(self.word2regtoken(words))
        return regtokens_list

    # 字转token
    def word2regtoken(self, words: list):
        regtokens = []
        regtokens.append("[CLS]")
        words_len = len(words)
        if words_len >= config.SENTENCE_LEN:
            regtokens.extend(words[:config.SENTENCE_LEN])
        else:
            regtokens.extend(words)
            for _ in range(config.SENTENCE_LEN - words_len):
                regtokens.append("[PAD]")
        regtokens.append("[SEP]")
        return regtokens

    # 标签序列处理 返回相同长度的标签序列
    def labels2reglabels(self, labels_list: list):
        logging.info("deal labels to reglabels")
        reglabels_list = list()
        for labels in labels_list:
            new_labels = list()
            # 配合token添加的[CLS]
            new_labels.append(self._total_labels[-1])
            labels_len = len(labels)
            if labels_len >= config.SENTENCE_LEN:
                new_labels.extend(labels[:config.SENTENCE_LEN])
            else:
                new_labels.extend(labels)
                for _ in range(labels_len - config.SENTENCE_LEN):
                    new_labels.append(self._total_labels[-1])
            # 配合token添加的[SEP]
            new_labels.append(self._total_labels[-1])
            reglabels_list.append(new_labels)
        return reglabels_list

    # 将tokens转为input_ids
    def tokens2input_ids(self, tokens_list: list):
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
    def labels2indexs(self, labels_list: list):
        logging.info("label to index")
        indexs_list = []  # 标签向量列表
        for labels in labels_list:
            indexs = []
            for label in labels:
                index = self.get_labelindex(label)
                indexs.append(index)
            indexs_list.append(indexs)

    # 处理标注数据
    def deal_tagdata(self, tagdata_filepaths: list, rate: float = config.WR_RATE):
        logging.info("begin deal word tag data")
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

        national_words_list, national_labels_list = self._split_tagdata(datas)
        words_list, labels_list = self.deal_wls(national_words_list, national_labels_list)
        national_words_list = None
        national_labels_list = None

        regtokens_list = self.words2regtokens(words_list)
        reglabels_list = self.labels2reglabels(labels_list)
        words_list = None
        labels_list = None

        input_ids = self.tokens2input_ids(regtokens_list)
        indexs_list = self.labels2indexs(reglabels_list)
        regtokens_list = None
        reglabels_list = None

        embeddings = self.input_ids2vec(input_ids)

        # 将数据保存下来
        total_size = len(embeddings)

        train_x = embeddings[:int(total_size * rate)]
        train_y = indexs_list[:int(total_size * rate)]
        test_x = embeddings[int(total_size * rate):]
        test_y = indexs_list[int(total_size * rate):]
        embeddings = None
        indexs_list = None

        logging.info("deal word tag data end")
        return train_x, train_y, test_x, test_y


preprocess = WordPreprocess()