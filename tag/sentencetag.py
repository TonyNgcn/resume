#!/usr/bin/python 
# -*- coding: utf-8 -*-

import logging

from tool import bigfile
from extract import splitsentence
from preprocess.bert_sentencepre import default_preprocess
from model.bert_sentencerec import BertSentenceRecModel
from tables.models import Mark
from db.mysql import Session


class SentenceRecTag(object):
    def __init__(self, model_name: str = "bert_sentencerec.ckpt"):
        self._model = BertSentenceRecModel(model_name=model_name, predictor=True)

    def tag(self, filepath: str):
        resumes = self._read_file(filepath)
        for resume in resumes:
            sentences = splitsentence.resume2sentences(resume)
            embeddings = default_preprocess.sentences2embeddings(sentences)
            labels = self._model.predict_label(embeddings)
            self._save(sentences, labels)

    # 读取简历文件
    def _read_file(self, filepath: str):
        resumes = []
        for line in bigfile.get_lines(filepath):
            resume = line.strip("\n")
            if resume:
                resumes.append(resume)
        return resumes

    # 保存标注数据
    def _save(self, sentences: list, labels: list):
        marks = []
        for sentence, label in zip(sentences, labels):
            mark = Mark(content=sentence, label_mark=label)
            marks.append(mark)
        sess = Session()
        try:
            sess.add_all(marks)
            sess.commit()
        except Exception as e:
            logging.error(e)


default_tag = SentenceRecTag()
