#!/usr/bin/python 
# -*- coding: utf-8 -*-

import logging
import numpy as np
import os

import config
from tool import shuffle, bigfile, mysplit
from holder.berth import bert_holder, tokenizer


# 命名实体识别模型数据加载类
class WordRecDataLoader(object):
    # get_traindata()和get_testdata()函数一定要有
    def __init__(self, rate: float = config.WR_RATE):
        if rate <= 0 or rate >= 1:
            logging.critical("rate must between 0 and 1")
        self._rate = rate

    def _get_data(self):
        datas = list()
        tagdata_filenames = os.listdir(config.TAG_WR_DIC)
        tagdata_filepaths = [config.TAG_WR_DIC + "/" + tagdata_filename
                             for tagdata_filename in tagdata_filenames if tagdata_filename != ".D_Store"]
        for tagdata_filepath in tagdata_filepaths:
            if os.path.exists(tagdata_filepath):
                for line in bigfile.get_lines(tagdata_filepath):
                    datas.append(line)
            else:
                logging.warning("tag data file {} is not exist".format(tagdata_filepath))
        return datas

    def _split_data(self, datas: list):
        if len(datas) % 2 == 1:
            logging.error("datas lenght error")

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

                    # words_list.append(words)
                    # labels_list.append(labels)

                    if len(words) == len(labels) and len(words) != 0:
                        words_list.append(words)
                        labels_list.append(labels)
                    else:
                        logging.warning("error data: {} {}".format(words, labels))

        return words_list, labels_list

    # 将实体词分字 实体标签也分
    def _deal_wls(self, national_words_list: list, national_labels_list: list):
        chars_list = []
        labels_list = []
        for national_words, national_labels in zip(national_words_list, national_labels_list):
            chars, labels = self._deal_wl(national_words, national_labels)
            chars_list.append(chars)
            labels_list.append(labels)
        return chars_list, labels_list

    # 将实体词分字 实体标签也分
    def _deal_wl(self, national_words: list, national_labels: list):
        chars = []
        labels = []
        for national_word, national_label in zip(national_words, national_labels):
            w = mysplit.split(national_word)
            # w = tokenizer.tokenize(national_word)
            # w = list(national_word)
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

    def get_data(self):
        datas = self._get_data()
        national_words_list, national_labels_list = self._split_data(datas)
        return self._deal_wls(national_words_list, national_labels_list)

    def get_traindata(self):
        datas = self._get_data()
        national_words_list, national_labels_list = self._split_data(datas)
        datas_len = len(national_words_list)
        traindata_len = int(datas_len * self._rate)
        return self._deal_wls(national_words_list[:traindata_len], national_labels_list[:traindata_len])

    def get_valdata(self):
        datas = self._get_data()
        national_words_list, national_labels_list = self._split_data(datas)
        datas_len = len(national_words_list)
        traindata_len = int(datas_len * self._rate)
        return self._deal_wls(national_words_list[traindata_len:int((traindata_len+datas_len)/2)],
                              national_labels_list[traindata_len:int((traindata_len+datas_len)/2)])

    def get_testdata(self):
        datas = self._get_data()
        national_words_list, national_labels_list = self._split_data(datas)
        datas_len = len(national_words_list)
        traindata_len = int(datas_len * self._rate)
        return self._deal_wls(national_words_list[int((traindata_len + datas_len) / 2):],
                              national_labels_list[int((traindata_len + datas_len) / 2):])


# 命名实体识别模型预处理类
class WordRecPreprocess(object):
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

    def fix(self, chars_list: list, labels_list: list):
        def labels_in(labels: str):
            for label in labels:
                if label not in self._total_labels:
                    return False
            return True

        fix_chars_list = []
        fix_labels_list = []
        for chars, labels in zip(chars_list, labels_list):
            chars_len = len(chars)
            labels_len = len(labels)
            if chars_len == labels_len and labels_in(labels):
                fix_chars_list.append(chars)
                fix_labels_list.append(labels)
            else:
                logging.warning("error data:chars:{} labels:{}".format(chars,labels))

        return fix_chars_list, fix_labels_list

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

    def _to_lower(self, chars_list: list):
        new_chars_list = []
        for chars in chars_list:
            new_chars = []
            for char in chars:
                if len(char) == 1:
                    char = char.lower()
                new_chars.append(char)
            new_chars_list.append(new_chars)
        return new_chars_list

    # 转成input_ids
    def _to_input_ids(self, chars_list: list):
        input_ids_list = []
        for chars in chars_list:
            logging.debug(chars)
            ids = tokenizer.convert_tokens_to_ids(chars)
            input_ids_list.append(ids)
        return input_ids_list

    # 转成embeddings
    def _to_embeddings(self, input_ids: list):
        embeddings = bert_holder.predict(input_ids)
        return embeddings

    def chars2embeddings(self, chars_list: list):
        regchars_list = self._regchar(chars_list)
        lower_chars_list = self._to_lower(regchars_list)
        input_ids_list = self._to_input_ids(lower_chars_list)
        embeddings = self._to_embeddings(input_ids_list)
        return embeddings

    def regchars2embeddings(self, regchars_list: list):
        lower_chars_list = self._to_lower(regchars_list)
        input_ids_list = self._to_input_ids(lower_chars_list)
        embeddings = self._to_embeddings(input_ids_list)
        return embeddings

    ###########################################################################
    # 截取标签
    def _reglabel(self, labels_list: list):
        reglabels_list = []
        for labels in labels_list:
            reglabels = []
            labels_len = len(labels)
            reglabels.append(self._total_labels[-1])
            if labels_len >= config.SENTENCE_LEN:
                reglabels.extend(labels[:config.SENTENCE_LEN])
            else:
                reglabels.extend(labels)
                for _ in range(config.SENTENCE_LEN - labels_len):
                    reglabels.append(self._total_labels[-1])
            reglabels.append(self._total_labels[-1])
            reglabels_list.append(reglabels)
        return reglabels_list

    # 标签转成索引值
    def _to_indexs(self, labels_list: list):
        indexs_list = []
        for labels in labels_list:
            indexs = []
            for label in labels:
                if label in self._total_labels:
                    index = self._total_labels.index(label)
                else:
                    logging.warning("label {} is not exist".format(label))
                    index = len(self._total_labels) - 1
                indexs.append(index)
            indexs_list.append(indexs)
        return indexs_list

    # 标签转成索引值
    def labels2indexs(self, labels_list: list):
        reglabels_list = self._reglabel(labels_list)
        indexs_list = self._to_indexs(reglabels_list)
        return indexs_list

    ###########################################################################
    # 获取标签列表
    def get_total_labels(self):
        return self._total_labels

    # 获取标签
    def get_labels(self, indexs: list):
        labels = []
        total_labels_len = len(self._total_labels)
        for index in indexs:
            if index < 0 or index >= total_labels_len:
                logging.error("index:{} out of range".format(index))
                label = self._total_labels[-1]
            else:
                label = self._total_labels[index]
            labels.append(label)
        return labels

    ###########################################################################
    # 加载训练数据
    def _load_traindata(self):
        try:
            wtrain_x = np.load(config.PREDATA_DIC + '/bert_wtrain_x.npy')
            wtrain_y = np.load(config.PREDATA_DIC + '/bert_wtrain_y.npy')
            return wtrain_x, wtrain_y
        except Exception as e:
            logging.error(e)
            exit(1)

    # 加载验证数据
    def _load_valdata(self):
        try:
            wval_x = np.load(config.PREDATA_DIC + '/bert_wval_x.npy')
            wval_y = np.load(config.PREDATA_DIC + '/bert_wval_y.npy')
            return wval_x, wval_y
        except Exception as e:
            logging.error(e)
            exit(1)

    # 加载测试数据
    def _load_testdata(self):
        try:
            wtest_x = np.load(config.PREDATA_DIC + '/bert_wtest_x.npy')
            wtest_y = np.load(config.PREDATA_DIC + '/bert_wtest_y.npy')
            return wtest_x, wtest_y
        except Exception as e:
            logging.error(e)
            exit(1)

    # 获取打乱后的训练数据
    def get_traindata(self):
        wtrain_x, wtrain_y = self._load_traindata()

        wtrain_x, wtrain_y = shuffle.shuffle_both(wtrain_x, wtrain_y)  # 打乱数据

        if len(wtrain_x) > 0:
            return wtrain_x, wtrain_y
        else:
            logging.error("train data length is less than 0")
            exit(1)

    # 获取打乱后的验证数据
    def get_valdata(self):
        wval_x, wval_y = self._load_valdata()

        wval_x, wval_y = shuffle.shuffle_both(wval_x, wval_y)  # 打乱数据

        if len(wval_x) > 0:
            return wval_x, wval_y
        else:
            logging.error("val data length is less than 0")
            exit(1)

    # 获取打乱后的测试数据
    def get_testdata(self):
        wtest_x, wtest_y = self._load_testdata()

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

    # 批量获取打乱后的验证数据
    def get_batch_valdata(self, batch_size: int):
        wval_x, wval_y = self.get_valdata()

        total_size = len(wval_x)
        start = 0
        while start + batch_size < total_size:
            yield wval_x[start:start + batch_size], wval_y[start:start + batch_size]
            start += batch_size
        if len(wval_x[start:]) > 0:
            yield wval_x[start:], wval_y[start:]

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
            os.remove(config.PREDATA_DIC + "/bert_wtrain_x.npy")
            os.remove(config.PREDATA_DIC + "/bert_wtrain_y.npy")
            logging.info("remove train data success")
        except Exception as e:
            logging.warning(e)

    # 删除测试数据
    def remove_valdata(self):
        try:
            os.remove(config.PREDATA_DIC + "/bert_wval_x.npy")
            os.remove(config.PREDATA_DIC + "/bert_wval_y.npy")
            logging.info("remove val data success")
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

    ###########################################################################
    # 处理标注的训练数据
    def deal_traindata(self, loader):
        chars_list, labels_list = loader.get_traindata()
        fix_chars_list, fix_labels_list = self.fix(chars_list, labels_list)
        embeddings = self.chars2embeddings(fix_chars_list)
        indexs_list = self.labels2indexs(fix_labels_list)
        self._save_data("bert_wtrain_x.npy", embeddings)
        self._save_data("bert_wtrain_y.npy", indexs_list)

    # 处理标注的测试数据
    def deal_valdata(self, loader):
        chars_list, labels_list = loader.get_valdata()
        fix_chars_list, fix_labels_list = self.fix(chars_list, labels_list)
        embeddings = self.chars2embeddings(fix_chars_list)
        indexs_list = self.labels2indexs(fix_labels_list)
        self._save_data("bert_wval_x.npy", embeddings)
        self._save_data("bert_wval_y.npy", indexs_list)

    # 处理标注的测试数据
    def deal_testdata(self, loader):
        chars_list, labels_list = loader.get_testdata()
        fix_chars_list, fix_labels_list = self.fix(chars_list, labels_list)
        embeddings = self.chars2embeddings(fix_chars_list)
        indexs_list = self.labels2indexs(fix_labels_list)
        self._save_data("bert_wtest_x.npy", embeddings)
        self._save_data("bert_wtest_y.npy", indexs_list)


default_preprocess = WordRecPreprocess()

if __name__ == '__main__':
    loader = WordRecDataLoader()
    loader.get_traindata()
    loader.get_valdata()
    loader.get_testdata()
    # preprocess = WordRecPreprocess()
    # preprocess.deal_traindata(loader)
    # preprocess.deal_traindata(loader)
