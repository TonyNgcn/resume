#!/usr/bin/python 
# -*- coding: utf-8 -*-

import os
import logging

import config
from model.bert_wordrec import BertWordRecModel
from extract import splitsentence
from tool import bigfile
from preprocess.bert_sentencepre import default_preprocess as spreprocess
from preprocess.bert_wordpre_test import default_preprocess as wpreprocess
from tables.models import WMark
from db.mysql import Session


class WordRecTag(object):
    def __init__(self, model_name: str = "bert_wordrec.ckpt"):
        self._model = BertWordRecModel(model_name=model_name, predictor=True)

    def tag(self, filepath: str):
        for resume in self._read_file(filepath):
            sentences = splitsentence.resume2sentences(resume)
            chars_list = spreprocess.sentences2chars_list(sentences)
            regchars_list = spreprocess.chars2regchars(chars_list)
            embeddings = wpreprocess.regchars2embeddings(regchars_list)
            labels_list = self._model.predict_label(embeddings)
            self._save(chars_list, regchars_list, labels_list)

    # 读取简历文件
    def _read_file(self, filepath: str):
        for line in bigfile.get_lines(filepath):
            resume = line.strip("\n")
            if resume:
                yield resume

    # 保存标注数据
    def _save(self, chars_list: list, regchars_list: list, labels_list: list):
        marks = []
        for chars, regchars, labels in zip(chars_list, regchars_list, labels_list):
            chars_len = len(chars)
            if chars_len <= config.SENTENCE_LEN:
                mark = WMark(content=" ".join(chars), label_marks=" ".join(labels[1:chars_len + 1]))
            else:
                mark = WMark(content=" ".join(chars[0:config.SENTENCE_LEN]),
                             label_marks=" ".join(labels[1:config.SENTENCE_LEN + 1]))
            marks.append(mark)
        sess = Session()
        try:
            sess.add_all(marks)
            sess.commit()
        except Exception as e:
            logging.error(e)


class CotrainWordRecTag(object):
    def __init__(self, model_name: str = "bert_wordrec.ckpt"):
        self._model = BertWordRecModel(model_name=model_name, predictor=True)

    def tag(self, filepath: str, suffix: str):
        filename = os.path.split(filepath)[-1]
        save_filepath = os.path.join(config.TMP_WR_DIC, suffix + "_" + filename)
        file = open(save_filepath, "w")
        for resume in self._read_file(filepath):
            sentences = splitsentence.resume2sentences(resume)
            chars_list = spreprocess.sentences2chars_list(sentences)
            regchars_list = spreprocess.chars2regchars(chars_list)
            embeddings = wpreprocess.regchars2embeddings(regchars_list)
            labels_list = self._model.predict_label(embeddings)
            self._save(chars_list, regchars_list, labels_list, file)
        file.close()

    # 读取简历文件
    def _read_file(self, filepath: str):
        for line in bigfile.get_lines(filepath):
            resume = line.strip("\n")
            if resume:
                yield resume

    # 保存标注数据
    def _save(self, chars_list: list, regchars_list: list, labels_list: list, file):
        for chars, regchars, labels in zip(chars_list, regchars_list, labels_list):
            chars_len = len(chars)
            if chars_len <= config.SENTENCE_LEN:
                str_chars = " ".join(chars)
                str_labels = " ".join(labels[1:chars_len + 1])
            else:
                str_chars = " ".join(chars[0:config.SENTENCE_LEN])
                str_labels = " ".join(labels[1:config.SENTENCE_LEN + 1])
            file.write(str_chars)
            file.write("\n")
            file.write(str_labels)
            file.write("\n")


default_tag = WordRecTag()
