#!/usr/bin/python 
# -*- coding: utf-8 -*-

import time
import json
import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from extract import splitsentence
from tool import bigfile
import config
from preprocess.bert_sentencepre import default_preprocess, SentenceRecDataLoader
from tool.evaluate import default_evaluate


class CotrainBertSentenceRecModel(object):
    def __init__(self, sentence_len: int = config.SENTENCE_LEN, wordvec_size: int = config.BERT_EMBEDDING_SIZE,
                 classes: int = len(default_preprocess.get_total_labels()),
                 study_rate: float = config.SR_STUDY_RATE, model_name: str = "cotrain_bert_sentencerec.ckpt",
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
              continue_train: bool = False, train_generator=None, val_generator=None, top_f1=0):
        logging.info("train sentence recognition model")
        if continue_train:
            sess, ph_x, ph_y, loss, train_opt, outputs = self.get_trained_model()
        else:
            sess, ph_x, ph_y, loss, train_opt, outputs = self.get_model()

        with sess.graph.as_default():
            saver = tf.train.Saver(max_to_keep=1)
            # 最高的f1值
            # 五次没有提高就停止训练
            top_p = top_r = top_f = 0
            # top_f1 = 0
            count = 0
            epoch = 1
            while True:
                train_true_y, train_pred_y = self._epoch_train(sess, ph_x, ph_y, train_opt, outputs, batch_size,
                                                               train_generator)
                acc = default_evaluate.calculate_accuracy(train_true_y, train_pred_y)
                precision, recall, f1 = default_evaluate.calculate_avg_prf(train_true_y, train_pred_y)
                print('epoch:{} batch size:{} acc:{} precision:{} recall:{} f1:{}'.format(epoch, batch_size, acc,
                                                                                          precision, recall, f1))

                test_true_y, test_pred_y, _ = self._epoch_val(sess, ph_x, ph_y, outputs, batch_size, val_generator)
                val_acc = default_evaluate.calculate_accuracy(test_true_y, test_pred_y)
                val_precision, val_recall, val_f1 = default_evaluate.calculate_avg_prf(test_true_y, test_pred_y)
                print('epoch:{} batch size:{} val_acc:{} val_precision:{} val_recall:{} val_f1:{}'.format(epoch,
                                                                                                          batch_size,
                                                                                                          val_acc,
                                                                                                          val_precision,
                                                                                                          val_recall,
                                                                                                          val_f1))
                if top_f1 < val_f1:
                    top_f1 = val_f1
                    top_p = val_precision
                    top_r = val_recall
                    top_f = val_f1
                    count = 0
                    logging.info("save sentencerec model")
                    saver.save(sess, config.MODEL_DIC + "/" + self._model_name)
                else:
                    if count >= 5:
                        return top_p, top_r, top_f
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
    def test(self, batch_size: int = config.SR_BATCH_SIZE, generator=None, limit=None):
        logging.info("test sentence recognition model")
        sess, ph_x, ph_y, loss, train_opt, outputs = self.get_model()

        with sess.graph.as_default():
            saver = tf.train.Saver(max_to_keep=1)
            saver.restore(sess, config.MODEL_DIC + "/" + self._model_name)

            test_true_y, test_pred_y, limit = self._epoch_test(sess, ph_x, ph_y, outputs, batch_size, generator, limit)
            print("测试数据长度:", len(test_true_y))
            time.sleep(5)
            default_evaluate.print_evaluate(test_true_y, test_pred_y, default_preprocess.get_total_labels())
            val_acc = default_evaluate.calculate_accuracy(test_true_y, test_pred_y)
            val_precision, val_recall, val_f1 = default_evaluate.calculate_avg_prf(test_true_y, test_pred_y)
            return limit, val_acc, val_precision, val_recall, val_f1

    # 单次迭代测试
    def _epoch_val(self, sess: tf.Session, ph_x, ph_y, pred, batch_size: int, generator=None, limit=None):
        total_true_y = None
        total_pred_y = None

        if generator is None:
            get_batch_valdata = default_preprocess.get_batch_valdata
        else:
            get_batch_valdata = generator

        for test_x, test_y in get_batch_valdata(batch_size):
            pred_y = sess.run(pred, feed_dict={ph_x: test_x, ph_y: test_y})

            index_true_y = np.argmax(test_y, axis=1).copy()
            index_pred_y = np.argmax(pred_y, axis=1).copy()

            if limit is not None:
                limit = self._fill(limit, pred_y, index_true_y, index_pred_y)

            if total_true_y is None:
                total_true_y = index_true_y
            else:
                total_true_y = np.concatenate([total_true_y, index_true_y])
            if total_pred_y is None:
                total_pred_y = index_pred_y
            else:
                total_pred_y = np.concatenate([total_pred_y, index_pred_y])

        return total_true_y, total_pred_y, limit

    # 单次迭代测试
    def _epoch_test(self, sess: tf.Session, ph_x, ph_y, pred, batch_size: int, generator=None, limit=None):
        total_true_y = None
        total_pred_y = None

        if generator is None:
            get_batch_testdata = default_preprocess.get_batch_testdata
        else:
            get_batch_testdata = generator

        for test_x, test_y in get_batch_testdata(batch_size):
            pred_y = sess.run(pred, feed_dict={ph_x: test_x, ph_y: test_y})

            index_true_y = np.argmax(test_y, axis=1).copy()
            index_pred_y = np.argmax(pred_y, axis=1).copy()

            if limit is not None:
                limit = self._fill(limit, pred_y, index_true_y, index_pred_y)

            if total_true_y is None:
                total_true_y = index_true_y
            else:
                total_true_y = np.concatenate([total_true_y, index_true_y])
            if total_pred_y is None:
                total_pred_y = index_pred_y
            else:
                total_pred_y = np.concatenate([total_pred_y, index_pred_y])

        return total_true_y, total_pred_y, limit

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

    # 预测 返回标签向量列表
    def predict_with_thresholds(self, inputs, sentences, thresholds):
        new_sentences = []
        new_labels = []
        with self._session.graph.as_default():
            pred_y = self._session.run(self._pred, feed_dict={self._ph_x: inputs})
            index_pred_y = np.argmax(pred_y, axis=1)
            for maxindex, vector, sentence in zip(index_pred_y, pred_y, sentences):
                label = default_preprocess.get_label(vector)
                p = vector[maxindex]
                for temp in thresholds[label]:
                    if temp["from"] <= p and temp["to"] >= p:
                        if temp["is_predict"]:
                            new_sentences.append(sentence)
                            new_labels.append(label)
                        break
        return new_sentences, new_labels

    def create_limit(self):
        labels = default_preprocess.get_total_labels()
        limit = {}
        interval = 0.025
        for label in labels:
            limit[label] = []
            down = 0
            while round(down + interval, 3) <= 1:
                temp = {"from": round(down, 3), "to": round(down + interval, 3), "true_data": [], "false_data": []}
                limit[label].append(temp)
                down += interval
        return limit

    def _fill(self, limit, pred_y, index_true_y, index_pred_y):
        labels = default_preprocess.get_total_labels()
        for true_index, pred_index, vector in zip(index_true_y, index_pred_y, pred_y):
            p = round(float(vector[true_index]), 3)
            label = labels[true_index]
            if true_index == pred_index:
                # 预测正确
                index = self._get_limit_index(limit, label, p)
                limit[label][index]["true_data"].append(p)
            else:
                # 预测错误
                index = self._get_limit_index(limit, label, p)
                limit[label][index]["false_data"].append(p)
        return limit

    def _get_limit_index(self, limit, label, p):
        r = limit[label]
        for index in range(len(r)):
            if (r[index]["from"] <= p) and (r[index]["to"] >= p):
                return index
        print(p)
        print(r)
        print(limit)
        raise IndexError("p out of range")

    def get_thresholds(self, limit):
        thresholds = {}
        for label in limit.keys():
            thresholds[label] = []
            for index in range(len(limit[label])):

                temp = {"from": limit[label][index]["from"], "to": limit[label][index]["to"]}
                # 判断该范围预测的数据是否可以用
                true_datas = limit[label][index]["true_data"]
                false_datas = limit[label][index]["false_data"]
                true_datas_len = len(true_datas)
                false_datas_len = len(false_datas)
                if (true_datas_len + false_datas_len) == 0:
                    temp["is_predict"] = False
                else:
                    if (true_datas_len > 0) and (true_datas_len * 1.0 / (true_datas_len + false_datas_len) >= 0.90):
                        temp["is_predict"] = True
                    else:
                        temp["is_predict"] = False
                thresholds[label].append(temp)
        return thresholds


########################################################################################


# 10000条数据作为训练集
# 5000条数据作为测试集
class CotrainDataLoader:
    def __init__(self, filename):
        self._filename = filename

    def _read_file(self):
        sentences = []
        labels = []
        for line in bigfile.get_lines(os.path.join(config.TAG_TMP_SR_DIC, self._filename)):
            line = line.strip("\n")
            pairs = line.split(";;;")
            sentences.append(pairs[0])
            labels.append(pairs[1])
        return sentences, labels

    def get_traindata(self):
        senteces, labels = self._read_file()
        return senteces, labels


class TestDataLoader:
    def _read_file(self):
        sentences = []
        labels = []
        count = 1
        for line in bigfile.get_lines(os.path.join(config.TMP_SR_DIC, "tag_sr3.txt")):
            print(line)
            print(count)
            count += 1
            line = line.strip("\n")
            pairs = line.split(";;;")
            sentences.append(pairs[0])
            labels.append(pairs[1])
        return sentences, labels

    def get_data(self):
        senteces, labels = self._read_file()
        return senteces, labels


class DataLoader(object):
    def __init__(self, embeddings, vectors):
        self._embeddings = embeddings
        self._vectors = vectors

    def get_batch_data(self, batch_size):
        total_size = len(self._embeddings)
        start = 0
        while start + batch_size < total_size:
            yield self._embeddings[start:start + batch_size], self._vectors[start:start + batch_size]
            start += batch_size
        if len(self._embeddings[start:]) > 0:
            yield self._embeddings[start:], self._vectors[start:]
        pass


class CotrainSentenceRecTag(object):
    def __init__(self, model_name: str = "cotrain_bert_sentencerec.ckpt"):
        self._model = CotrainBertSentenceRecModel(model_name=model_name, predictor=True)

    def tag(self, filepath: str, thresholds):
        filename = os.path.split(filepath)[-1]
        file = open(config.TAG_TMP_SR_DIC + "/" + filename, "w")
        for resume in self._read_file(filepath):
            sentences = splitsentence.resume2sentences(resume)
            embeddings = default_preprocess.sentences2embeddings(sentences)
            new_sentences, new_labels = self._model.predict_with_thresholds(embeddings, sentences, thresholds)
            self._save(new_sentences, new_labels, file)
        file.close()

    # 读取简历文件
    def _read_file(self, filepath: str):
        for line in bigfile.get_lines(filepath):
            resume = line.strip("\n")
            if resume:
                yield resume

    # 保存标注数据
    def _save(self, sentences: list, labels: list, file):
        for sentence, label in zip(sentences, labels):
            file.write(sentence)
            file.write(";;;")
            file.write(label)
            file.write("\n")


def get_test_loader():
    test_loader = TestDataLoader()
    sentences, labels = test_loader.get_data()
    test_embeddings = default_preprocess.sentences2embeddings(sentences)
    test_vectors = default_preprocess.labels2vectors(labels)

    test_data_loader = DataLoader(test_embeddings, test_vectors)
    return test_data_loader


def cotrain():
    loader = SentenceRecDataLoader(rate=0.2)
    # preprocess = SentenceRecPreprocess()
    default_preprocess.deal_valdata(loader)
    default_preprocess.deal_testdata(loader)
    default_preprocess.deal_traindata(loader)

    png_epochs = []
    png_accs = []
    png_f1s = []
    top_f1 = 0

    limit_file = open("fill_limits.json", "w")

    log = open("cotrain.log", "w")

    # test_data_loader = get_test_loader()

    model = CotrainBertSentenceRecModel()
    model.train(train_generator=default_preprocess.get_batch_traindata,
                val_generator=default_preprocess.get_batch_valdata)
    # 获取阈值
    fill_limit, top_acc, _, _, top_f1 = model.test(limit=model.create_limit(),
                                                   generator=default_preprocess.get_batch_testdata)
    limit_file.write(json.dumps(fill_limit, ensure_ascii=False))
    limit_file.write("\n")
    limit_file.flush()

    print("没有进行迭代训练时的测试数据f1值：", top_f1)
    log.write("没有进行迭代训练时的测试数据f1值：{}\n".format(top_f1))
    thresholds = model.get_thresholds(fill_limit)

    # _, val_acc, val_precision, val_recall, val_f1 = model.test(generator=test_data_loader.get_batch_data)
    # print("没有进行迭代训练时的验证数据f1值：", val_f1)
    # log.write("没有进行迭代训练时的验证数据f1值：{}\n".format(val_f1))

    file = open("thresholds.json", "w")
    file.write(json.dumps(thresholds, ensure_ascii=False))
    file.close()

    count = 0
    epoch = 1
    png_epochs.append(epoch)
    png_accs.append(top_acc)
    png_f1s.append(top_f1)

    filenames = os.listdir(config.SRCDATA_DIC)
    for filename in filenames:
        if filename not in ["resume_data1.txt", "resume_data2.txt", "resume_data3.txt", ".D_Store"]:
            # 使用训练好的模型预测未标注数据
            stag = CotrainSentenceRecTag()
            stag.tag(os.path.join(config.SRCDATA_DIC, filename), thresholds)
            # 把预测出来的未标注数据作为训练数据，迭代训练模型
            train_loader = CotrainDataLoader(filename)
            sentences, labels = train_loader.get_traindata()
            if len(sentences) > 0:
                embeddings = default_preprocess.sentences2embeddings(sentences)
                vectors = default_preprocess.labels2vectors(labels)

                data_loader = DataLoader(embeddings, vectors)

                model.train(continue_train=True, train_generator=data_loader.get_batch_data,
                            val_generator=default_preprocess.get_batch_valdata, top_f1=top_f1)

                fill_limit, acc, _, _, f1 = model.test(limit=model.create_limit(),
                                                       generator=default_preprocess.get_batch_testdata)
                limit_file.write(json.dumps(fill_limit, ensure_ascii=False))
                limit_file.write("\n")
                limit_file.flush()

                print("第{}次进行迭代训练时的测试数据f1值：{}".format(epoch, f1))
                log.write("第{}次进行迭代训练时的测试数据f1值：{}\n".format(epoch, f1))

                thresholds = model.get_thresholds(fill_limit)

                # _, val_acc, val_precision, val_recall, val_f1 = model.test(generator=test_data_loader.get_batch_data)
                # print("第{}次进行迭代训练时的验证数据f1值：{}".format(epoch, val_f1))
                # log.write("第{}次进行迭代训练时的验证数据f1值：{}\n".format(epoch, val_f1))

                file = open("thresholds.json", "w")
                file.write(json.dumps(thresholds, ensure_ascii=False))
                file.close()

                png_epochs.append(epoch)
                png_accs.append(acc)
                png_f1s.append(f1)

                epoch += 1

                if f1 >= top_f1:
                    top_f1 = f1
                    count = 0
                else:
                    count += 1
                    if count >= 5:
                        plt.figure(figsize=(10, 5))
                        plt.title("Training text classification model using semi-supervised learning")
                        plt.xlabel(u"Epoch")
                        plt.xticks(np.arange(0, len(png_epochs) + 1, 1))
                        plt.ylabel(u"Effect")
                        plt.yticks(np.arange(0, 1.1, 0.05))
                        plt.plot(png_epochs, png_accs, "-", label="Accuracy")
                        plt.plot(png_epochs, png_f1s, "-", color="r", label="F1-measure")
                        plt.legend()
                        plt.grid()
                        plt.savefig(config.PNG_DIC + "/简历特征句分类模型半监督学习实验.png")

                        print("迭代训练完成")
                        break

                log.flush()
    log.close()
    limit_file.close()
