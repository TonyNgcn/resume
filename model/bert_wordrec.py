#!/usr/bin/python 
# -*- coding: utf-8 -*-

import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib import crf
import matplotlib.pyplot as plt

import config
from preprocess.bert_wordpre_test import default_preprocess
from tool.evaluate import default_evaluate


# BERT+BiGRU+CNN+CRF
class BertWordRecModel(object):
    def __init__(self, sentence_len: int = config.SENTENCE_LEN, wordvec_size: int = config.BERT_EMBEDDING_SIZE,
                 classes: int = len(default_preprocess.get_total_labels()),
                 study_rate: float = config.WR_STUDY_RATE, model_name: str = "bert_wordrec.ckpt",
                 predictor: bool = False):
        self._sentence_len = sentence_len
        self._wordvec_size = wordvec_size
        self._classes = classes
        self._study_rate = study_rate
        self._model_name = model_name
        if predictor:
            self._session, self._ph_sequence_lengths, self._ph_x, _, _, _, self._pred = self.get_trained_model()

    # 获取模型的图
    def get_model(self):
        graph = tf.Graph()
        session = tf.Session(graph=graph)
        with session.graph.as_default():
            ph_x = tf.placeholder(dtype=tf.float32, shape=[None, self._sentence_len + 2,
                                                           self._wordvec_size])  # shape(bactch_size,sentence_len,wordvec_size)
            ph_y = tf.placeholder(dtype=tf.int32,
                                  shape=[None, self._sentence_len + 2])  # shape(bactch_size,sentence_len)
            ph_sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[None, ])

            embeddings = keras.layers.Dense(config.WORDVEC_SIZE)(ph_x)

            # mask = keras.layers.Masking(mask_value=0.)(ph_x)
            bigru = keras.layers.Bidirectional(
                keras.layers.GRU(200, return_sequences=True))(embeddings)
            bigru = keras.layers.Dropout(0.5)(bigru)

            half_window_size = 2
            padding_layer = keras.layers.ZeroPadding1D(padding=half_window_size)(embeddings)
            conv = keras.layers.Conv1D(100, 2 * half_window_size + 1)(padding_layer)
            conv_d = keras.layers.Dropout(0.5)(conv)
            dense_conv = keras.layers.TimeDistributed(keras.layers.Dense(100))(conv_d)

            rnn_cnn = tf.concat([bigru, dense_conv], axis=2)

            dense = keras.layers.Dense(self._classes)(rnn_cnn)
            unary_scores = keras.layers.Dropout(0.5)(dense)

            log_likelihood, transition_params = crf.crf_log_likelihood(unary_scores, ph_y, ph_sequence_lengths)
            loss = tf.reduce_mean(-log_likelihood)

            viterbi_sequence, viterbi_score = crf.crf_decode(unary_scores, transition_params, ph_sequence_lengths)

            train_opt = tf.train.AdamOptimizer(self._study_rate).minimize(loss)

            init = tf.global_variables_initializer()
            session.run(init)

            return session, ph_sequence_lengths, ph_x, ph_y, loss, train_opt, viterbi_sequence

    # 获取训练好的模型
    def get_trained_model(self):
        sess, ph_sequence_lengths, ph_x, ph_y, loss, train_opt, viterbi_sequence = self.get_model()
        try:
            with sess.graph.as_default():
                saver = tf.train.Saver(max_to_keep=1)
                saver.restore(sess, config.MODEL_DIC + "/" + self._model_name)
        except Exception as e:
            logging.error(e)
            exit(1)
        return sess, ph_sequence_lengths, ph_x, ph_y, loss, train_opt, viterbi_sequence

    # 训练模型
    def train(self, batch_size: int = config.WR_BATCH_SIZE,
              continue_train: bool = False, train_generator=None, val_generator=None):
        logging.info("wordrec model train")
        if continue_train:
            sess, ph_sequence_lengths, ph_x, ph_y, loss, train_opt, outputs = self.get_trained_model()
        else:
            sess, ph_sequence_lengths, ph_x, ph_y, loss, train_opt, outputs = self.get_model()

        with sess.graph.as_default():
            saver = tf.train.Saver(max_to_keep=1)
            # 最高的f1值
            # 五次没有提高就停止训练
            top_f1 = 0
            count = 0
            epoch = 1

            png_epochs = []
            png_accs = []
            png_f1s = []

            plt.figure(figsize=(10, 5))
            plt.title("Training a named entity recognition model using BERT+BiGRU+CNN+CRF")
            plt.xlabel(u"Epoch")
            plt.xticks(np.arange(0, config.WR_EPOCHS + 1, 1))
            plt.ylabel(u"Effect")
            plt.yticks(np.arange(0, 1.1, 0.05))

            while True:
                train_true_y, train_pred_y = self._epoch_train(sess, ph_sequence_lengths, ph_x, ph_y, train_opt,
                                                               outputs, batch_size, train_generator)
                acc = default_evaluate.calculate_accuracy(train_true_y, train_pred_y)
                precision, recall, f1 = default_evaluate.calculate_avg_prf(train_true_y, train_pred_y)
                print(
                    'epoch:{} batch size:{} acc:{} precision:{} recall:{} f1:{}'.format(epoch, batch_size, acc,
                                                                                        precision, recall, f1))

                test_true_y, test_pred_y = self._epoch_val(sess, ph_sequence_lengths, ph_x, ph_y, outputs,
                                                            batch_size, val_generator)
                val_acc = default_evaluate.calculate_accuracy(train_true_y, train_pred_y)
                val_precision, val_recall, val_f1 = default_evaluate.calculate_avg_prf(test_true_y, test_pred_y)
                print('epoch:{} batch size:{} val_acc:{} val_precision:{} val_recall:{} val_f1:{}'.format(epoch,
                                                                                                          batch_size,
                                                                                                          val_acc,
                                                                                                          val_precision,
                                                                                                          val_recall,
                                                                                                          val_f1))

                png_epochs.append(epoch)
                png_accs.append(val_acc)
                png_f1s.append(val_f1)

                if top_f1 < val_f1:
                    top_f1 = val_f1
                    count = 0
                    logging.info("save wordrec model")
                    saver.save(sess, config.MODEL_DIC + "/" + self._model_name)
                # else:
                #     if count >= 5:
                #         plt.plot(png_epochs, png_accs, "-", label="Accuracy")
                #         plt.plot(png_epochs, png_f1s, "-", color="r", label="F1-measure")
                #         plt.savefig(config.PNG_DIC + "/BERT命名实体识别模型.png")
                #         break
                #     count += 1

                if epoch >= config.WR_EPOCHS:
                    plt.plot(png_epochs, png_accs, "-", label="Accuracy")
                    plt.plot(png_epochs, png_f1s, "-", color="r", label="F1-measure")
                    plt.legend()
                    plt.grid()
                    plt.savefig(config.PNG_DIC + "/BERT+BiGRU+CNN+CRF命名实体识别模型.png")
                    break
                epoch += 1

    # 单次迭代训练
    def _epoch_train(self, sess: tf.Session, ph_sequence_lengths, ph_x, ph_y, train_opt, pred, batch_size: int,
                     generator=None):
        total_true_y = None
        total_pred_y = None

        if generator is None:
            get_batch_traindata = default_preprocess.get_batch_traindata
        else:
            get_batch_traindata = generator

        for train_x, train_y in get_batch_traindata(batch_size):
            lengths = [self._sentence_len for _ in range(len(train_y))]
            sequence_lengths = np.array(lengths, dtype=np.int32)
            _, pred_y = sess.run([train_opt, pred],
                                 feed_dict={ph_sequence_lengths: sequence_lengths, ph_x: train_x,
                                            ph_y: train_y})

            true_y = np.reshape(train_y, [-1, ]).copy()
            pred_y = np.reshape(pred_y, [-1, ]).copy()

            if total_true_y is None:
                total_true_y = true_y
            else:
                total_true_y = np.concatenate([total_true_y, true_y], axis=None)
            if total_pred_y is None:
                total_pred_y = pred_y
            else:
                total_pred_y = np.concatenate([total_pred_y, pred_y], axis=None)
        return total_true_y, total_pred_y

    # 测试模型
    def test(self, batch_size: int = config.SR_BATCH_SIZE, generator=None):
        logging.info("wordrec model test")
        sess, ph_sequence_lengths, ph_x, ph_y, loss, train_opt, outputs = self.get_model()
        with sess.graph.as_default():
            saver = tf.train.Saver(max_to_keep=1)
            saver.restore(sess, config.MODEL_DIC + "/" + self._model_name)

            test_true_y, test_pred_y = self._epoch_test(sess, ph_sequence_lengths, ph_x, ph_y, outputs, batch_size,
                                                        generator)
            print(test_true_y, test_pred_y, default_preprocess.get_total_labels())
            default_evaluate.print_evaluate(test_true_y, test_pred_y, default_preprocess.get_total_labels())

    # 单次迭代测试
    def _epoch_test(self, sess: tf.Session, ph_sequence_lengths, ph_x, ph_y, pred, batch_size: int, generator=None):
        total_true_y = None
        total_pred_y = None

        if generator is None:
            get_batch_testdata = default_preprocess.get_batch_testdata
        else:
            get_batch_testdata = generator

        for test_x, test_y in get_batch_testdata(batch_size):
            lengths = [self._sentence_len for _ in range(len(test_y))]
            sequence_lengths = np.array(lengths, dtype=np.int32)
            pred_y = sess.run(pred,
                              feed_dict={ph_sequence_lengths: sequence_lengths, ph_x: test_x, ph_y: test_y})

            true_y = np.reshape(test_y, [-1, ]).copy()
            pred_y = np.reshape(pred_y, [-1, ]).copy()

            if total_true_y is None:
                total_true_y = true_y
            else:
                total_true_y = np.concatenate([total_true_y, true_y], axis=None)
            if total_pred_y is None:
                total_pred_y = pred_y
            else:
                total_pred_y = np.concatenate([total_pred_y, pred_y], axis=None)
        return total_true_y, total_pred_y

    # 单次迭代测试
    def _epoch_val(self, sess: tf.Session, ph_sequence_lengths, ph_x, ph_y, pred, batch_size: int, generator=None):
        total_true_y = None
        total_pred_y = None

        if generator is None:
            get_batch_valdata = default_preprocess.get_batch_valdata
        else:
            get_batch_valdata = generator

        for val_x, val_y in get_batch_valdata(batch_size):
            lengths = [self._sentence_len for _ in range(len(val_y))]
            sequence_lengths = np.array(lengths, dtype=np.int32)
            pred_y = sess.run(pred,
                              feed_dict={ph_sequence_lengths: sequence_lengths, ph_x: val_x, ph_y: val_y})

            true_y = np.reshape(val_y, [-1, ]).copy()
            pred_y = np.reshape(pred_y, [-1, ]).copy()

            if total_true_y is None:
                total_true_y = true_y
            else:
                total_true_y = np.concatenate([total_true_y, true_y], axis=None)
            if total_pred_y is None:
                total_pred_y = pred_y
            else:
                total_pred_y = np.concatenate([total_pred_y, pred_y], axis=None)
        return total_true_y, total_pred_y

    # 预测 返回标签向量列表
    def predict(self, inputs):
        with self._session.graph.as_default():
            lengths = [self._sentence_len for _ in range(len(inputs))]
            sequence_lengths = np.array(lengths, dtype=np.int32)
            pred_y = self._session.run(self._pred,
                                       feed_dict={self._ph_sequence_lengths: sequence_lengths, self._ph_x: inputs})
            return pred_y

    # 预测 返回标签列表
    def predict_label(self, inputs):
        pred_y = self.predict(inputs)
        labels_list = list()
        for indexs in pred_y:
            labels = default_preprocess.get_labels(indexs)
            labels_list.append(labels)
        return labels_list


default_model = BertWordRecModel()
