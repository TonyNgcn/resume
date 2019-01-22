#!/usr/bin/python 
# -*- coding: utf-8 -*-

import os
import logging
from pymongo import MongoClient

import config
from extract.extract import Extractor

logging.info("启动简历抽取器")
extractor = Extractor()

client = MongoClient("mongodb://127.0.0.1:27017/")
db = client["regresume_db"]
collection = db["regresume"]

# 获取简历
filenames = os.listdir(config.SRCDATA_DIC)
filepaths = [config.SRCDATA_DIC + "/" + filename for filename in filenames]

resumes = []
count = 0

for filepath in filepaths:
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            line_index = 0
            for line in file:
                line_index += 1
                resumes.append(line)
                count += 1

                if count >= 10:
                    regresumes = extractor.batch_extract(resumes)
                    try:
                        collection.insert_many(regresumes)
                    except:
                        logging.error(
                            "insert error from {} to {} in {}".format(line_index - len(resumes), line_index, filepath))
                    resumes.clear()
                    count = 0

if count > 0:
    regresumes = extractor.batch_extract(resumes)
    try:
        collection.insert_many(regresumes)
    except:
        pass
    resumes.clear()
    count = 0
