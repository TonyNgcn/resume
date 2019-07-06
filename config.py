#!/usr/bin/python 
# -*- coding: utf-8 -*-

# 配置文件

import os
import logging

# 目录路径
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))  # 获取项目根目录

SRCDATA_DIC = PROJECT_ROOT + "/file/srcdata"  # 简历源数据目录

PREDATA_DIC = PROJECT_ROOT + "/file/predata"  # 中间文件目录

CORPUS_DIC = PROJECT_ROOT + "/file/corpus"  # 文集文件目录

MODEL_DIC = PROJECT_ROOT + "/file/model"  # 模型文件目录

LOG_DIC = PROJECT_ROOT + "/file/log"  # 日志文件目录

TAG_DIC = PROJECT_ROOT + "/file/tag"  # 标注数据文件目录
TAG_SR_DIC = TAG_DIC + "/sr"
TAG_WR_DIC = TAG_DIC + "/wr"

COTRAIN_DIC = PROJECT_ROOT + "/file/cotrain"  # 协同训练模型标注比对后相同的数据文件目录
TAG_TMP_SR_DIC = COTRAIN_DIC + "/sr"
TAG_TMP_WR_DIC = COTRAIN_DIC + "/wr"

TMP_DIC = PROJECT_ROOT + "/file/tmp"  # 协同训练模型未处理数据文件目录
TMP_SR_DIC = TMP_DIC + "/sr"
TMP_WR_DIC = TMP_DIC + "/wr"

PNG_DIC = PROJECT_ROOT + "/file/png"

# 模型配置
SENTENCE_LEN = 50  # 25  # 句子长度

BERT_EMBEDDING_SIZE = 768

WORDVEC_SIZE = 150  # 120  # 词向量维度

# 特征句分类模型参数

SR_RATE = 0.80  # 训练集占标注集的百分比

SR_STUDY_RATE = 0.001  # 学习率

SR_EPOCHS = 16  # 训练模型迭代次数

SR_BATCH_SIZE = 200  # 训练模型每次输入数据条数

SR_BIGRU_UNITS = 400  # bigru 单元个数
SR_BIGRU_DROPOUT = 0.5  # bigru dropout值

# 命名实体识别模型参数

WR_RATE = 0.70  # 训练集占标注集的百分比

WR_STUDY_RATE = 0.005  # 学习率

WR_EPOCHS = 50  # 训练模型迭代次数

WR_BATCH_SIZE = 200  # 训练模型每次输入数据条数

WR_BIGRU_UNITS = 256
WR_BIGRU_DROPOUT = 0.5

WR_BIGRU1_UNITS = 256  # bigru1 单元个数
WR_BIGRU1_DROPOUT = 0.5  # bigru1 dropout值

# logging.basicConfig(filename=LOG_DIC+"/resume_import.log",format='%(asctime)s %(levelname)s:%(message)s', level=logging.ERROR)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.WARNING)

# 数据库
DIALECT = 'mysql'
DRIVER = 'pymysql'
USERNAME = 'root'
PASSWORD = 'root'
HOST = '127.0.0.1'
PORT = '3306'
DATABASE = 'mark_data'

SQLALCHEMY_DATABASE_URI = '{}+{}://{}:{}@{}:{}/{}?charset=utf8'.format(DIALECT, DRIVER, USERNAME, PASSWORD, HOST, PORT,
                                                                       DATABASE)
SQLALCHEMY_TRACK_MODIFICATIONS = False
