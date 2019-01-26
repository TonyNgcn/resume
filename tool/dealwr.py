#!/usr/bin/python 
# -*- coding: utf-8 -*-

import config
import logging

total_labels = ['B-name', 'I-name', 'E-name',
                'B-sex', 'I-sex', 'E-sex',
                'B-nationality', 'I-nationality', 'E-nationality',
                'B-nation', 'I-nation', 'E-nation',
                'B-time', 'I-time', 'E-time',
                'B-school', 'I-school', 'E-school',
                # 'B-college', 'I-college', 'E-college',
                'B-pro', 'I-pro', 'E-pro',
                'B-degree', 'I-degree', 'E-degree',
                'B-edu', 'I-edu', 'E-edu',
                'B-company', 'I-company', 'E-company',
                'B-department', 'I-department', 'E-department',
                'B-job', 'I-job', 'E-job',
                'O']  # 标签列表

total_nation_labels = [
    "name", "sex", "nationality", "nation", "time", "school", "pro", "degree", "edu", "company", "department", "job",
    "O"]


# 输入一条词序列，一条标签序列
# 输出实体词序列，实体词标签
def to_nation(words: list, labels: list):
    if len(words) != len(labels):
        return [], []
    nation_words = []
    nation_labels = []
    name = []
    sex = []
    nationality = []
    nation = []
    time = []
    school = []
    pro = []
    degree = []
    edu = []
    company = []
    department = []
    job = []
    o = []
    for word, label in zip(words, labels):
        if label == 'B-name':
            name.clear()
            name.append(word)
        elif label == 'I-name':
            name.append(word)
        elif label == 'E-name':
            if len(name) > 0:
                name.append(word)
                nation_words.append(''.join(name))
                nation_labels.append('name')
                name.clear()
            else:
                nation_words.append(word)
                nation_labels.append('name')
        ###########################################
        if label == 'B-sex':
            sex.clear()
            sex.append(word)
        elif label == 'I-sex':
            sex.append(word)
        elif label == 'E-sex':
            if len(sex) > 0:
                sex.append(word)
                nation_words.append(''.join(sex))
                nation_labels.append('sex')
                sex.clear()
            else:
                nation_words.append(word)
                nation_labels.append('sex')
        ###########################################
        if label == 'B-nationality':
            nationality.clear()
            nationality.append(word)
        elif label == 'I-nationality':
            nationality.append(word)
        elif label == 'E-nationality':
            if len(nationality) > 0:
                nationality.append(word)
                nation_words.append(''.join(nationality))
                nation_labels.append('nationality')
                nationality.clear()
            else:
                nation_words.append(word)
                nation_labels.append('nationality')
        ###########################################
        if label == 'B-nation':
            nation.clear()
            nation.append(word)
        elif label == 'I-nation':
            nation.append(word)
        elif label == 'E-nation':
            if len(nation) > 0:
                nation.append(word)
                nation_words.append(''.join(nation))
                nation_labels.append('nation')
                nation.clear()
            else:
                nation_words.append(word)
                nation_labels.append('nation')
        ###########################################
        if label == 'B-time':
            time.clear()
            time.append(word)
        elif label == 'I-time':
            time.append(word)
        elif label == 'E-time':
            if len(time) > 0:
                time.append(word)
                nation_words.append(''.join(time))
                nation_labels.append('time')
                time.clear()
            else:
                nation_words.append(word)
                nation_labels.append('time')
        ###########################################
        if label == 'B-school':
            school.clear()
            school.append(word)
        elif label == 'I-school':
            school.append(word)
        elif label == 'E-school':
            if len(school) > 0:
                school.append(word)
                nation_words.append(''.join(school))
                nation_labels.append('school')
                school.clear()
            else:
                nation_words.append(word)
                nation_labels.append('school')
        ###########################################
        if label == 'B-pro':
            pro.clear()
            pro.append(word)
        elif label == 'I-pro':
            pro.append(word)
        elif label == 'E-pro':
            if len(pro) > 0:
                pro.append(word)
                nation_words.append(''.join(pro))
                nation_labels.append('pro')
                pro.clear()
            else:
                nation_words.append(word)
                nation_labels.append('pro')
        ###########################################
        if label == 'B-degree':
            degree.clear()
            degree.append(word)
        elif label == 'I-degree':
            degree.append(word)
        elif label == 'E-degree':
            if len(degree) > 0:
                degree.append(word)
                nation_words.append(''.join(degree))
                nation_labels.append('degree')
                degree.clear()
            else:
                nation_words.append(word)
                nation_labels.append('degree')
        ###########################################
        if label == 'B-edu':
            edu.clear()
            edu.append(word)
        elif label == 'I-edu':
            edu.append(word)
        elif label == 'E-edu':
            if len(edu) > 0:
                edu.append(word)
                nation_words.append(''.join(edu))
                nation_labels.append('edu')
                edu.clear()
            else:
                nation_words.append(word)
                nation_labels.append('edu')
        ###########################################
        if label == 'B-company':
            company.clear()
            company.append(word)
        elif label == 'I-company':
            company.append(word)
        elif label == 'E-company':
            if len(company) > 0:
                company.append(word)
                nation_words.append(''.join(company))
                nation_labels.append('company')
                company.clear()
            else:
                nation_words.append(word)
                nation_labels.append('company')
        ###########################################
        if label == 'B-department':
            department.clear()
            department.append(word)
        elif label == 'I-department':
            department.append(word)
        elif label == 'E-department':
            if len(department) > 0:
                department.append(word)
                nation_words.append(''.join(department))
                nation_labels.append('department')
                department.clear()
            else:
                nation_words.append(word)
                nation_labels.append('department')
        ###########################################
        if label == 'B-job':
            job.clear()
            job.append(word)
        elif label == 'I-job':
            job.append(word)
        elif label == 'E-job':
            if len(job) > 0:
                job.append(word)
                nation_words.append(''.join(job))
                nation_labels.append('job')
                job.clear()
            else:
                nation_words.append(word)
                nation_labels.append('job')
        ###########################################
        if label == "O":
            nation_words.append(word)
            nation_labels.append("O")
    return nation_words, nation_labels


def to_nations(words_list: list, labels_list: list):
    nation_words_list = []
    nation_labels_list = []
    for words, labels in zip(words_list, labels_list):
        nation_words, nation_labels = to_nation(words, labels)
        nation_words_list.append(nation_words)
        nation_labels_list.append(nation_labels)
    return nation_words_list, nation_labels_list


def _split_tagdata(datas: list):
    if len(datas) % 2 == 1:
        logging.error("datas lenght error")

    global total_labels

    def wrong_words(words):
        for word in words:
            if word in total_labels:
                return True
        return False

    def wrong_labels(labels):
        for label in labels:
            if label not in total_labels:
                return True
        return False

    words_list = list()  # 保存分词后的句子 每一项是字符串： 词 词
    labels_list = list()  # 保存标签 每一项是字符串： label label

    # 奇数行是分好词的句子，偶数行是对应的词标签
    single = True  # 判断是奇数行还是偶数行 True 为奇数行
    words = list()

    for line in datas:
        if line:
            if single:
                words = line.strip("\n").split(" ")
                # 要判断是否乱行
                single = False
            else:
                labels = line.strip("\n").split(" ")
                single = True

                if len(words) == len(labels) and len(words) != 0 and not wrong_words(words) and not wrong_labels(
                        labels):
                    words_list.append(words)
                    labels_list.append(labels)
                else:
                    logging.warning("error data: {} {}".format(words, labels))

    return words_list, labels_list


if __name__ == '__main__':
    read_file = open(config.TAG_WR_DIC + "/tag_wr1.txt", "r")
    write_file = open(config.TMP_WR_DIC + "/tag_wr1.txt", "w")

    datas = []
    for line in read_file:
        print(line)
        datas.append(line)
    read_file.close()
    words_list, labels_list = _split_tagdata(datas)
    print(words_list,labels_list)
    nation_words_list, nation_labels_list = to_nations(words_list, labels_list)
    print(nation_words_list,nation_labels_list)
    for nation_words, nation_labels in zip(nation_words_list, nation_labels_list):
        write_file.write(" ".join(nation_words))
        write_file.write("\n")
        write_file.write(" ".join(nation_labels))
        write_file.write("\n")
    write_file.close()
