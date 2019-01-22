#!/usr/bin/python 
# -*- coding: utf-8 -*-

from model.sentencerec import SentenceRecModel
from model.wordrec import WordRecModel


class Predictor(object):
    def __init__(self, smodel_name: str = "sentencerec.ckpt", wmodel_name: str = "wordrec.ckpt"):
        self._smodel = SentenceRecModel(model_name=smodel_name, predictor=True)
        self._wmodel = WordRecModel(model_name=wmodel_name, predictor=True)

    def sentence_predict(self, inputs):
        return self._smodel.predict_label(inputs)

    def word_predict(self, inputs):
        return self._wmodel.predict_label(inputs)
