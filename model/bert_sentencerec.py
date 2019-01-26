#!/usr/bin/python 
# -*- coding: utf-8 -*-

import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
from preprocess.bert_sentencepre import default_preprocess
from tool.evaluate import default_evaluate


class BertSentenceRecModel(object):
    def __init__(self, sentence_len: int = config.SENTENCE_LEN, wordvec_size: int = config.BERT_EMBEDDING_SIZE,
                 classes: int = len(default_preprocess.get_total_labels()),
                 study_rate: float = config.SR_STUDY_RATE, model_name: str = "bert_sentencerec.ckpt",
                 predictor: bool = False):
        self._sentence_len = sentence_len
        self._wordvec_size = wordvec_size
        self._classes = classes
        self._study_rate = study_rate
        self._model_name = model_name
        if predictor:
            self._session, self._ph_x, _, _, _, self._pred = self.get_trained_model()

    # 获取模型的图
    def get_model(self):
        logging.info("get sentence recognition model")
        graph = tf.Graph()
        with graph.as_default():
            ph_x = tf.placeholder(dtype=tf.float32, shape=[None, self._sentence_len + 2,
                                                           self._wordvec_size])  # shape(bactch_size,sentence_len+2,wordvec_size)
            ph_y = tf.placeholder(dtype=tf.float32,
                                  shape=[None, self._classes])  # shape(bactch_size,classifi_size)

            dense = keras.layers.Dense(config.WORDVEC_SIZE)(ph_x)

            bigru = keras.layers.Bidirectional(
                keras.layers.GRU(config.SR_BIGRU_UNITS, return_sequences=False, dropout=config.SR_BIGRU_DROPOUT))(dense)
            outputs = keras.layers.Dense(self._classes, activation=tf.nn.softmax)(bigru)

            cross = keras.losses.categorical_crossentropy(y_true=ph_y, y_pred=outputs)
            loss = tf.reduce_mean(cross)

            train_opt = tf.train.AdamOptimizer(self._study_rate).minimize(loss)
            init = tf.global_variables_initializer()

        sess = tf.Session(graph=graph)
        sess.run(init)

        return sess, ph_x, ph_y, loss, train_opt, outputs

    # 获取训练好的模型
    def get_trained_model(self):
        logging.info("get trained sentence recognition model")
        sess, ph_x, ph_y, loss, train_opt, pred = self.get_model()
        try:
            with sess.graph.as_default():
                saver = tf.train.Saver(max_to_keep=1)
                saver.restore(sess, config.MODEL_DIC + "/" + self._model_name)
        except Exception as e:
            logging.error(e)
            exit(1)
        return sess, ph_x, ph_y, loss, train_opt, pred

    # 训练模型
    def train(self, batch_size: int = config.SR_BATCH_SIZE,
              continue_train: bool = False, train_generator=None, test_generator=None):
        logging.info("train sentence recognition model")
        if continue_train:
            sess, ph_x, ph_y, loss, train_opt, outputs = self.get_trained_model()
        else:
            sess, ph_x, ph_y, loss, train_opt, outputs = self.get_model()

        with sess.graph.as_default():
            saver = tf.train.Saver(max_to_keep=1)
            # 最高的f1值
            # 五次没有提高就停止训练
            top_f1 = 0
            count = 0
            epoch = 1
            while True:
                train_true_y, train_pred_y = self._epoch_train(sess, ph_x, ph_y, train_opt, outputs, batch_size,
                                                               train_generator)
                acc = default_evaluate.calculate_accuracy(train_true_y, train_pred_y)
                precision, recall, f1 = default_evaluate.calculate_avg_prf(train_true_y, train_pred_y)
                print('epoch:{} batch size:{} acc:{} precision:{} recall:{} f1:{}'.format(epoch, batch_size, acc,
                                                                                          precision, recall, f1))

                test_true_y, test_pred_y = self._epoch_test(sess, ph_x, ph_y, outputs, batch_size, test_generator)
                val_acc = default_evaluate.calculate_accuracy(test_true_y, test_pred_y)
                val_precision, val_recall, val_f1 = default_evaluate.calculate_avg_prf(test_true_y, test_pred_y)
                print('epoch:{} batch size:{} val_acc:{} val_precision:{} val_recall:{} val_f1:{}'.format(epoch,
                                                                                                          batch_size,
                                                                                                          val_acc,
                                                                                                          val_precision,
                                                                                                          val_recall,
                                                                                                          val_f1))
                if top_f1 <= val_f1:
                    top_f1 = val_f1
                    count = 0
                    logging.info("save sentencerec model")
                    saver.save(sess, config.MODEL_DIC + "/" + self._model_name)
                else:
                    if count >= 5:
                        break
                    count += 1
                epoch += 1

    # 单次迭代训练
    def _epoch_train(self, sess: tf.Session, ph_x, ph_y, train_opt, pred, batch_size: int, generator=None):
        total_true_y = None
        total_pred_y = None

        if generator is None:
            get_batch_traindata = default_preprocess.get_batch_traindata
        else:
            get_batch_traindata = generator

        for train_x, train_y in get_batch_traindata(batch_size):
            _, train_pred_y = sess.run([train_opt, pred], feed_dict={ph_x: train_x, ph_y: train_y})

            true_y = np.argmax(train_y, axis=1).copy()
            pred_y = np.argmax(train_pred_y, axis=1).copy()

            if total_true_y is None:
                total_true_y = true_y
            else:
                total_true_y = np.concatenate([total_true_y, true_y])
            if total_pred_y is None:
                total_pred_y = pred_y
            else:
                total_pred_y = np.concatenate([total_pred_y, pred_y])
        return total_true_y, total_pred_y

    # 测试模型
    def test(self, batch_size: int = config.SR_BATCH_SIZE, generator=None):
        logging.info("test sentence recognition model")
        sess, ph_x, ph_y, loss, train_opt, outputs = self.get_model()

        with sess.graph.as_default():
            saver = tf.train.Saver(max_to_keep=1)
            saver.restore(sess, config.MODEL_DIC + "/" + self._model_name)

            test_true_y, test_pred_y = self._epoch_test(sess, ph_x, ph_y, outputs, batch_size, generator)
            print(test_pred_y, test_true_y, default_preprocess.get_total_labels())
            default_evaluate.print_evaluate(test_true_y, test_pred_y, default_preprocess.get_total_labels())

    # 单次迭代测试
    def _epoch_test(self, sess: tf.Session, ph_x, ph_y, pred, batch_size: int, generator=None):
        total_true_y = None
        total_pred_y = None

        if generator is None:
            get_batch_testdata = default_preprocess.get_batch_testdata
        else:
            get_batch_testdata = generator

        for test_x, test_y in get_batch_testdata(batch_size):
            pred_y = sess.run(pred, feed_dict={ph_x: test_x, ph_y: test_y})

            true_y = np.argmax(test_y, axis=1).copy()
            pred_y = np.argmax(pred_y, axis=1).copy()

            if total_true_y is None:
                total_true_y = true_y
            else:
                total_true_y = np.concatenate([total_true_y, true_y])
            if total_pred_y is None:
                total_pred_y = pred_y
            else:
                total_pred_y = np.concatenate([total_pred_y, pred_y])

        return total_true_y, total_pred_y

    # 预测 返回标签向量列表
    def predict(self, inputs):
        with self._session.graph.as_default():
            pred_y = self._session.run(self._pred, feed_dict={self._ph_x: inputs})
            return pred_y

    # 预测 返回标签列表
    def predict_label(self, inputs):
        pred_y = self.predict(inputs)
        labels = list()
        for vector in pred_y:
            labels.append(default_preprocess.get_label(vector))
        return labels


defualt_model = BertSentenceRecModel()
