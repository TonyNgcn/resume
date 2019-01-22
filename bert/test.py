#!/usr/bin/python 
# -*- coding: utf-8 -*-

from .modeling import BertConfig, BertModel
from .tokenization import FullTokenizer
import tensorflow as tf

batch_size = 120
seq_length = 30


def main():
    text = "廖建政，女，1966年4月出生，中国籍，无境外永久居留权，研究生学历。1990年3月至1994年2月在湖南师大附中任教英语；1994年3月至1997年2月，在土耳其正大集团欧洲区担任统计工作；1997年3月至1999年1月，在湖南师大广益中学任教英语；1999年2月至2002年4月，在家待业；2002年5月至2015年5月，任有限公司监事；2015年5月至今，在家待业。"
    # preprocess
    tokenizer = FullTokenizer(vocab_file="chinese_L-12_H-768_A-12/vocab.txt")
    tokens = []
    for i, word in enumerate(text):
        if i < 30:
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            print(token)
    print("tokens", tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input = [input_ids]
    print("input_ids", input_ids)
    # model
    graph = tf.Graph()
    with graph.as_default():
        ph_input_ids = tf.placeholder(dtype=tf.int32, shape=[None, seq_length], name="ph_input_ids")
        # ph_input_mask = tf.placeholder(dtype=tf.int32, shape=[batch_size, seq_length], name="ph_input_mask")
        # ph_token_type_ids = tf.placeholder(dtype=tf.int32, shape=[batch_size, seq_length], name="ph_token_type_ids")
        config = BertConfig.from_json_file("chinese_L-12_H-768_A-12/bert_config.json")
        bert_model = BertModel(config=config, is_training=False, input_ids=ph_input_ids, use_one_hot_embeddings=True)
        # bert_model = BertModel(config=config, is_training=False, input_ids=ph_input_ids, input_mask=ph_input_mask,
        #                        token_type_ids=ph_token_type_ids, use_one_hot_embeddings=True)
        output = bert_model.get_sequence_output()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        with tf.Session(graph=graph) as sess:
            sess.run(init)
            saver.restore(sess, "chinese_L-12_H-768_A-12/bert_model.ckpt")
            print(output.shape)
            # embeddings shape []
            embeddings = sess.run(output, feed_dict={ph_input_ids: input})

            print(embeddings)


if __name__ == '__main__':
    main()
