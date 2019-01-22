#!/usr/bin/python 
# -*- coding: utf-8 -*-

import config
from model.mybert import MyBertModel
from bert.tokenization import FullTokenizer

if __name__ == '__main__':
    bert_model = MyBertModel()
    text = "廖建政，女，1966年4月出生，中国籍，无境外永久居留权，研究生学历。1990年3月至1994年2月在湖南师大附中任教英语；1994年3月至1997年2月，在土耳其正大集团欧洲区担任统计工作；1997年3月至1999年1月，在湖南师大广益中学任教英语；1999年2月至2002年4月，在家待业；2002年5月至2015年5月，任有限公司监事；2015年5月至今，在家待业。"
    tokenizer = FullTokenizer(vocab_file=config.CORPUS_DIC + "/vocab.txt")
    tokens = []
    tokens.append("[CLS]")
    for i, word in enumerate(text):
        if i < config.SENTENCE_LEN:
            token = tokenizer.tokenize(word)
            print("token:",token)
            tokens.extend(token)
    tokens.append("[SEP]")
    print(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(input_ids)
    inputs = [input_ids]
    embeddings = bert_model.predict(inputs)
    print(embeddings)
