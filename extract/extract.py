#!/usr/bin/python 
# -*- coding: utf-8 -*-

from .predict import BertPredictor
from .splitsentence import resume2sentences
# from preprocess.sentencepre import preprocess as srpre

from preprocess.bert_sentencepre import default_preprocess as spreprocess
from preprocess.bert_wordpre_test import default_preprocess as wpreprocess


class ResumeFormat(object):
    # 简历信息输出
    def regresume_print(self, regresume: dict):
        print('basic info:')
        print('  name:', regresume['basic_info']['name'])
        print('  sex:', regresume['basic_info']['sex'])
        print('  nationality:', regresume['basic_info']['nationality'])
        print('  nation:', regresume['basic_info']['nation'])
        print('  born:', regresume['basic_info']['born'])
        print('school exp:')
        print(' past:')
        for exp in regresume['school_exp']['past']:
            print('  time:', exp['time'])
            print('  school:', exp['school'])
            # print('  college:', exp['college'])
            print('  pro:', exp['pro'])
            print('  degree:', exp['degree'])
            print('  edu:', exp['edu'])
            print()
        print(' current:')
        for exp in regresume['school_exp']['current']:
            print('  time:', exp['time'])
            print('  school:', exp['school'])
            # print('  college:', exp['college'])
            print('  pro:', exp['pro'])
            print('  degree:', exp['degree'])
            print('  edu:', exp['edu'])
            print()
        print('work exp')
        print(' past:')
        for exp in regresume['work_exp']['past']:
            print('  time:', exp['time'])
            print('  place:', exp['place'])
            print('  department:', exp['department'])
            print('  job:', exp['job'])
            print()
        print(' current:')
        for exp in regresume['work_exp']['current']:
            print('  time:', exp['time'])
            print('  place:', exp['place'])
            print('  department:', exp['department'])
            print('  job:', exp['job'])
            print()

    # 简历有用信息输出
    def regresume_regprint(self, regresume: dict):
        print('basic info:')
        print('  name:', regresume['basic_info']['name'])
        print('  sex:', regresume['basic_info']['sex'])
        if regresume['basic_info']['nationality'] != '':
            print('  nationality:', regresume['basic_info']['nationality'])
        if regresume['basic_info']['nation'] != '':
            print('  nation:', regresume['basic_info']['nation'])
        if regresume['basic_info']['born'] != '':
            print('  born:', regresume['basic_info']['born'])
        print('school exp:')
        print(' past:')
        for exp in regresume['school_exp']['past']:
            if exp['time'] != '':
                print('  time:', exp['time'])
            if exp['school'] != '':
                print('  school:', exp['school'])
            # if exp['college'] != '':
            #     print('  college:', exp['college'])
            if exp['pro'] != '':
                print('  pro:', exp['pro'])
            if exp['degree'] != '':
                print('  degree:', exp['degree'])
            if exp['edu'] != '':
                print('  edu:', exp['edu'])
            print()
        print(' current:')
        for exp in regresume['school_exp']['current']:
            if exp['time'] != '':
                print('  time:', exp['time'])
            if exp['school'] != '':
                print('  school:', exp['school'])
            # if exp['college'] != '':
            #     print('  college:', exp['college'])
            if exp['pro'] != '':
                print('  pro:', exp['pro'])
            if exp['degree'] != '':
                print('  degree:', exp['degree'])
            if exp['edu'] != '':
                print('  edu:', exp['edu'])
            print()
        print('work exp')
        print(' past:')
        for exp in regresume['work_exp']['past']:
            if exp['time'] != '':
                print('  time:', exp['time'])
            if exp['place'] != '':
                print('  place:', exp['place'])
            if exp['job'] != '':
                print('  job:', exp['job'])
            print()
        print(' current:')
        for exp in regresume['work_exp']['current']:
            if exp['time'] != '':
                print('  time:', exp['time'])
            if exp['place'] != '':
                print('  place:', exp['place'])
            if exp['job'] != '':
                print('  job:', exp['job'])
            print()

    # 简历信息转成str
    def regresume2str(self, regresume: dict):
        msg = []
        msg.append('basic info:')
        msg.append('  name: ' + regresume['basic_info']['name'])
        msg.append('  sex: ' + regresume['basic_info']['sex'])
        msg.append('  nationality: ' + regresume['basic_info']['nationality'])
        msg.append('  nation: ' + regresume['basic_info']['nation'])
        msg.append('  born: ' + regresume['basic_info']['born'])
        msg.append('school exp:')
        msg.append(' past:')
        for exp in regresume['school_exp']['past']:
            msg.append('  time: ' + exp['time'])
            msg.append('  school: ' + exp['school'])
            # msg.append('  college: ' + exp['college'])
            msg.append('  pro: ' + exp['pro'])
            msg.append('  degree: ' + exp['degree'])
            msg.append('  edu: ' + exp['edu'])
            msg.append('\n')
        msg.append(' current:')
        for exp in regresume['school_exp']['current']:
            msg.append('  time: ' + exp['time'])
            msg.append('  school: ' + exp['school'])
            # msg.append('  college: ' + exp['college'])
            msg.append('  pro: ' + exp['pro'])
            msg.append('  degree: ' + exp['degree'])
            msg.append('  edu: ' + exp['edu'])
            msg.append('\n')
        msg.append('work exp')
        msg.append(' past:')
        for exp in regresume['work_exp']['past']:
            msg.append('  time: ' + exp['time'])
            msg.append('  place: ' + exp['place'])
            msg.append('  job: ' + '、'.join(exp['job']))
            msg.append('\n')
        msg.append(' current:')
        for exp in regresume['work_exp']['current']:
            msg.append('  time: ' + exp['time'])
            msg.append('  place: ' + exp['place'])
            msg.append('  job: ' + '、'.join(exp['job']))
            msg.append('\n')

        return '\n'.join(msg)

    # 简历有用信息转成str
    def regresume2regstr(self, regresume: dict):
        msg = []
        msg.append('basic info:')
        msg.append('  name: ' + regresume['basic_info']['name'])
        msg.append('  sex: ' + regresume['basic_info']['sex'])
        if regresume['basic_info']['nationality'] != '':
            msg.append('  nationality: ' + regresume['basic_info']['nationality'])
        if regresume['basic_info']['nation'] != '':
            msg.append('  nation: ' + regresume['basic_info']['nation'])
        if regresume['basic_info']['born'] != '':
            msg.append('  born: ' + regresume['basic_info']['born'])
        msg.append('school exp:')
        msg.append(' past:')
        for exp in regresume['school_exp']['past']:
            if exp['time'] != '':
                msg.append('  time: ' + exp['time'])
            if exp['school'] != '':
                msg.append('  school: ' + exp['school'])
            # if exp['college'] != '':
            #     msg.append('  college: ' + exp['college'])
            if exp['pro'] != '':
                msg.append('  pro: ' + exp['pro'])
            if exp['degree'] != '':
                msg.append('  degree: ' + exp['degree'])
            if exp['edu'] != '':
                msg.append('  edu: ' + exp['edu'])
            msg.append('\n')
        msg.append(' current:')
        for exp in regresume['school_exp']['current']:
            if exp['time'] != '':
                msg.append('  time: ' + exp['time'])
            if exp['school'] != '':
                msg.append('  school: ' + exp['school'])
            # if exp['college'] != '':
            #     msg.append('  college: ' + exp['college'])
            if exp['pro'] != '':
                msg.append('  pro: ' + exp['pro'])
            if exp['degree'] != '':
                msg.append('  degree: ' + exp['degree'])
            if exp['edu'] != '':
                msg.append('  edu: ' + exp['edu'])
            msg.append('\n')
        msg.append('work exp')
        msg.append(' past:')
        for exp in regresume['work_exp']['past']:
            if exp['time'] != '':
                msg.append('  time: ' + exp['time'])
            if exp['place'] != '':
                msg.append('  place: ' + exp['place'])
            if exp['job'] != '':
                msg.append('  job: ' + exp['job'])
            msg.append('\n')
        msg.append(' current:')
        for exp in regresume['work_exp']['current']:
            if exp['time'] != '':
                msg.append('  time: ' + exp['time'])
            if exp['place'] != '':
                msg.append('  place: ' + exp['place'])
            if exp['job'] != '':
                msg.append('  job: ' + exp['job'])
            msg.append('\n')

        return '\n'.join(msg)


default_format = ResumeFormat()


class Extractor(object):
    def __init__(self):
        self._predictor = BertPredictor()

    def single_extract(self, resume: str):
        return self._extract(resume)

    def batch_extract(self, resumes: list):
        regresumes = []
        for resume in resumes:
            regresumes.append(self._extract(resume))
        return regresumes

    # 批量简历抽取生成器
    def batch_extract_generator(self, resumes: list):
        for resume in resumes:
            yield self._extract(resume)

    def _extract(self, resume: str):
        sentences = resume2sentences(resume)

        regchars_list = spreprocess.sentences2regchars_list(sentences)
        embeddings = wpreprocess.regchars2embeddings(regchars_list)

        sentence_label_list = self._predictor.sentence_predict(embeddings)
        word_labels_list = self._predictor.word_predict(embeddings)

        embeddings = None

        # 处理实体词
        sentence_label_list, notional_words_list, notional_labels_list = self._deal_notional_words(
            sentence_label_list, regchars_list, word_labels_list)

        # 将实体词转成格式化简历
        return self._notional_words2regresume(sentence_label_list, notional_words_list, notional_labels_list)

    # 处理实体词
    def _deal_notional_words(self, sentence_label_list: list, words_list: list, word_labels_list: list):
        notional_words_list = []  # 保存实体词
        notional_labels_list = []  # 保存实体词标签

        total_labels = spreprocess.get_total_labels()
        for sentence_label, words, word_labels in zip(sentence_label_list, words_list, word_labels_list):
            if sentence_label == total_labels[0]:  # ctime
                notional_words, notional_labels = self._deal_ctime(words, word_labels)
            elif sentence_label == total_labels[1]:  # ptime
                notional_words, notional_labels = self._deal_ptime(words, word_labels)
            elif sentence_label == total_labels[2]:  # basic
                notional_words, notional_labels = self._deal_basic(words, word_labels)
            elif sentence_label == total_labels[3]:  # wexp
                notional_words, notional_labels = self._deal_wexp(words, word_labels)
            elif sentence_label == total_labels[4]:  # sexp
                notional_words, notional_labels = self._deal_sexp(words, word_labels)
            elif sentence_label == total_labels[5]:  # noinfo
                notional_words, notional_labels = self._deal_noinfo()
            else:  # error 出错
                raise Exception('该句子标签不存在')

            notional_words_list.append(notional_words)
            notional_labels_list.append(notional_labels)

        # 返回 每个句子的标签和每个句子的实体词标签列表和实体词列表
        return sentence_label_list, notional_words_list, notional_labels_list

    # 处理ctime标签信息
    def _deal_ctime(self, words: list, word_labels: list):
        notional_words = []
        notional_labels = []

        time = []
        school = []
        college = []
        pro = []
        degree = []
        edu = []
        company = []
        department = []
        job = []

        for word, word_label in zip(words, word_labels):
            if word_label == 'B-time':
                time.clear()
                time.append(word)
            elif word_label == 'I-time':
                time.append(word)
            elif word_label == 'E-time':
                if len(time) > 0:
                    time.append(word)
                    notional_words.append(''.join(time))
                    notional_labels.append('time')
                    time.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('time')
            #################################################
            elif word_label == 'B-school':
                school.clear()
                school.append(word)
            elif word_label == 'I-school':
                school.append(word)
            elif word_label == 'E-school':
                if len(school) > 0:
                    school.append(word)
                    notional_words.append(''.join(school))
                    notional_labels.append('school')
                    school.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('school')
            #################################################
            # elif word_label == 'B-college':
            #     college.clear()
            #     college.append(word)
            # elif word_label == 'I-college':
            #     college.append(word)
            # elif word_label == 'E-college':
            #     if len(college) > 0:
            #         college.append(word)
            #         notional_words.append(''.join(college))
            #         notional_labels.append('college')
            #         college.clear()
            #     else:
            #         notional_words.append(word)
            #         notional_labels.append('college')
            #################################################
            elif word_label == 'B-pro':
                pro.clear()
                pro.append(word)
            elif word_label == 'I-pro':
                pro.append(word)
            elif word_label == 'E-pro':
                if len(pro) > 0:
                    pro.append(word)
                    notional_words.append(''.join(pro))
                    notional_labels.append('pro')
                    pro.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('pro')
            #################################################
            elif word_label == 'B-degree':
                degree.clear()
                degree.append(word)
            elif word_label == 'I-degree':
                degree.append(word)
            elif word_label == 'E-degree':
                if len(degree) > 0:
                    degree.append(word)
                    notional_words.append(''.join(degree))
                    notional_labels.append('degree')
                    degree.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('degree')
            #################################################
            elif word_label == 'B-edu':
                edu.clear()
                edu.append(word)
            elif word_label == 'I-edu':
                edu.append(word)
            elif word_label == 'E-edu':
                if len(edu) > 0:
                    edu.append(word)
                    notional_words.append(''.join(edu))
                    notional_labels.append('edu')
                    edu.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('edu')
            #################################################
            elif word_label == 'B-company':
                company.clear()
                company.append(word)
            elif word_label == 'I-company':
                company.append(word)
            elif word_label == 'E-company':
                if len(company) > 0:
                    company.append(word)
                    notional_words.append(''.join(company))
                    notional_labels.append('company')
                    company.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('company')
            #################################################
            elif word_label == 'B-department':
                department.clear()
                department.append(word)
            elif word_label == 'I-department':
                department.append(word)
            elif word_label == 'E-department':
                if len(department) > 0:
                    department.append(word)
                    notional_words.append(''.join(department))
                    notional_labels.append('department')
                    department.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('department')
            #################################################
            elif word_label == 'B-job':
                job.clear()
                job.append(word)
            elif word_label == 'I-job':
                job.append(word)
            elif word_label == 'E-job':
                if len(job) > 0:
                    job.append(word)
                    notional_words.append(''.join(job))
                    notional_labels.append('job')
                    job.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('job')
            #################################################

        return notional_words, notional_labels

        # 处理ptime标签信息

    def _deal_ptime(self, words: list, word_labels: list):
        notional_words = []
        notional_labels = []

        time = []
        school = []
        # college = []
        pro = []
        degree = []
        edu = []
        company = []
        department = []
        job = []

        for word, word_label in zip(words, word_labels):
            if word_label == 'B-time':
                time.clear()
                time.append(word)
            elif word_label == 'I-time':
                time.append(word)
            elif word_label == 'E-time':
                if len(time) > 0:
                    time.append(word)
                    notional_words.append(''.join(time))
                    notional_labels.append('time')
                    time.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('time')
            #################################################
            elif word_label == 'B-school':
                school.clear()
                school.append(word)
            elif word_label == 'I-school':
                school.append(word)
            elif word_label == 'E-school':
                if len(school) > 0:
                    school.append(word)
                    notional_words.append(''.join(school))
                    notional_labels.append('school')
                    school.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('school')
            #################################################
            # elif word_label == 'B-college':
            #     college.clear()
            #     college.append(word)
            # elif word_label == 'I-college':
            #     college.append(word)
            # elif word_label == 'E-college':
            #     if len(college) > 0:
            #         college.append(word)
            #         notional_words.append(''.join(college))
            #         notional_labels.append('college')
            #         college.clear()
            #     else:
            #         notional_words.append(word)
            #         notional_labels.append('college')
            #################################################
            elif word_label == 'B-pro':
                pro.clear()
                pro.append(word)
            elif word_label == 'I-pro':
                pro.append(word)
            elif word_label == 'E-pro':
                if len(pro) > 0:
                    pro.append(word)
                    notional_words.append(''.join(pro))
                    notional_labels.append('pro')
                    pro.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('pro')
            #################################################
            elif word_label == 'B-degree':
                degree.clear()
                degree.append(word)
            elif word_label == 'I-degree':
                degree.append(word)
            elif word_label == 'E-degree':
                if len(degree) > 0:
                    degree.append(word)
                    notional_words.append(''.join(degree))
                    notional_labels.append('degree')
                    degree.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('degree')
            #################################################
            elif word_label == 'B-edu':
                edu.clear()
                edu.append(word)
            elif word_label == 'I-edu':
                edu.append(word)
            elif word_label == 'E-edu':
                if len(edu) > 0:
                    edu.append(word)
                    notional_words.append(''.join(edu))
                    notional_labels.append('edu')
                    edu.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('edu')
            #################################################
            elif word_label == 'B-company':
                company.clear()
                company.append(word)
            elif word_label == 'I-company':
                company.append(word)
            elif word_label == 'E-company':
                if len(company) > 0:
                    company.append(word)
                    notional_words.append(''.join(company))
                    notional_labels.append('company')
                    company.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('company')
            #################################################
            elif word_label == 'B-department':
                department.clear()
                department.append(word)
            elif word_label == 'I-department':
                department.append(word)
            elif word_label == 'E-department':
                if len(department) > 0:
                    department.append(word)
                    notional_words.append(''.join(department))
                    notional_labels.append('department')
                    department.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('department')
            #################################################
            elif word_label == 'B-job':
                job.clear()
                job.append(word)
            elif word_label == 'I-job':
                job.append(word)
            elif word_label == 'E-job':
                if len(job) > 0:
                    job.append(word)
                    notional_words.append(''.join(job))
                    notional_labels.append('job')
                    job.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('job')
            #################################################

        return notional_words, notional_labels

        # 处理basic标签信息

    def _deal_basic(self, words: list, word_labels: list):
        notional_words = []
        notional_labels = []

        name = []
        sex = []
        nationality = []
        nation = []
        time = []

        for word, word_label in zip(words, word_labels):
            if word_label == 'B-name':
                name.clear()
                name.append(word)
            elif word_label == 'I-name':
                name.append(word)
            elif word_label == 'E-name':
                if len(name) > 0:
                    name.append(word)
                    notional_words.append(''.join(name))
                    notional_labels.append('name')
                    name.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('name')
            #################################################
            elif word_label == 'B-sex':
                sex.clear()
                sex.append(word)
            elif word_label == 'I-sex':
                sex.append(word)
            elif word_label == 'E-sex':
                if len(sex) > 0:
                    sex.append(word)
                    notional_words.append(''.join(sex))
                    notional_labels.append('sex')
                    name.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('sex')
            #################################################
            elif word_label == 'B-nationality':
                nationality.clear()
                nationality.append(word)
            elif word_label == 'I-nationality':
                nationality.append(word)
            elif word_label == 'E-nationality':
                if len(nationality) > 0:
                    nationality.append(word)
                    notional_words.append(''.join(nationality))
                    notional_labels.append('nationality')
                    nationality.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('nationality')
            #################################################
            elif word_label == 'B-nation':
                nation.clear()
                nation.append(word)
            elif word_label == 'I-nation':
                nation.append(word)
            elif word_label == 'E-nation':
                if len(nation) > 0:
                    nation.append(word)
                    notional_words.append(''.join(nation))
                    notional_labels.append('nation')
                    nation.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('nation')
            #################################################
            elif word_label == 'B-time':
                time.clear()
                time.append(word)
            elif word_label == 'I-time':
                time.append(word)
            elif word_label == 'E-time':
                if len(time) > 0:
                    time.append(word)
                    notional_words.append(''.join(time))
                    notional_labels.append('time')
                    time.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('time')
            #################################################

        return notional_words, notional_labels

        # 处理wexp标签信息

    def _deal_wexp(self, words: list, word_labels: list):
        notional_words = []
        notional_labels = []

        school = []
        company = []
        department = []
        job = []

        for word, word_label in zip(words, word_labels):
            if word_label == 'B-school':
                school.clear()
                school.append(word)
            elif word_label == 'I-school':
                school.append(word)
            elif word_label == 'E-school':
                if len(school) > 0:
                    school.append(word)
                    notional_words.append(''.join(school))
                    notional_labels.append('school')
                    school.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('school')
            #################################################
            elif word_label == 'B-company':
                company.clear()
                company.append(word)
            elif word_label == 'I-company':
                company.append(word)
            elif word_label == 'E-company':
                if len(company) > 0:
                    company.append(word)
                    notional_words.append(''.join(company))
                    notional_labels.append('company')
                    company.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('company')
            #################################################
            elif word_label == 'B-department':
                department.clear()
                department.append(word)
            elif word_label == 'I-department':
                department.append(word)
            elif word_label == 'E-department':
                if len(department) > 0:
                    department.append(word)
                    notional_words.append(''.join(department))
                    notional_labels.append('department')
                    department.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('department')
            #################################################
            elif word_label == 'B-job':
                job.clear()
                job.append(word)
            elif word_label == 'I-job':
                job.append(word)
            elif word_label == 'E-job':
                if len(job) > 0:
                    job.append(word)
                    notional_words.append(''.join(job))
                    notional_labels.append('job')
                    job.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('job')
            #################################################

        return notional_words, notional_labels

        # 处理sexp标签信息

    def _deal_sexp(self, words: list, word_labels: list):
        notional_words = []
        notional_labels = []

        school = []
        # college = []
        pro = []
        degree = []
        edu = []

        for word, word_label in zip(words, word_labels):
            if word_label == 'B-school':
                school.clear()
                school.append(word)
            elif word_label == 'I-school':
                school.append(word)
            elif word_label == 'E-school':
                if len(school) > 0:
                    school.append(word)
                    notional_words.append(''.join(school))
                    notional_labels.append('school')
                    school.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('school')
            #################################################
            # elif word_label == 'B-college':
            #     college.clear()
            #     college.append(word)
            # elif word_label == 'I-college':
            #     college.append(word)
            # elif word_label == 'E-college':
            #     if len(college) > 0:
            #         college.append(word)
            #         notional_words.append(''.join(college))
            #         notional_labels.append('college')
            #         college.clear()
            #     else:
            #         notional_words.append(word)
            #         notional_labels.append('college')
            #################################################
            elif word_label == 'B-pro':
                pro.clear()
                pro.append(word)
            elif word_label == 'I-pro':
                pro.append(word)
            elif word_label == 'E-pro':
                if len(pro) > 0:
                    pro.append(word)
                    notional_words.append(''.join(pro))
                    notional_labels.append('pro')
                    pro.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('pro')
            #################################################
            elif word_label == 'B-degree':
                degree.clear()
                degree.append(word)
            elif word_label == 'I-degree':
                degree.append(word)
            elif word_label == 'E-degree':
                if len(degree) > 0:
                    degree.append(word)
                    notional_words.append(''.join(degree))
                    notional_labels.append('degree')
                    degree.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('degree')
            #################################################
            elif word_label == 'B-edu':
                edu.clear()
                edu.append(word)
            elif word_label == 'I-edu':
                edu.append(word)
            elif word_label == 'E-edu':
                if len(edu) > 0:
                    edu.append(word)
                    notional_words.append(''.join(edu))
                    notional_labels.append('edu')
                    edu.clear()
                else:
                    notional_words.append(word)
                    notional_labels.append('edu')
            #################################################

        return notional_words, notional_labels

        # 处理noinfo标签信息

    def _deal_noinfo(self):
        notional_words = []
        notional_labels = []

        return notional_words, notional_labels

        # 实体词转成格式化简历信息

    def _notional_words2regresume(self, sentence_label_list: list, notional_words_list: list,
                                  notional_labels_list: list):
        # 保存格式化的简历
        regresume = {
            'basic_info': {
                'name': '',
                'sex': '',
                'nationality': '',
                'nation': '',
                'born': ''
            },

            'school_exp': {
                'past': [
                    # {
                    #     'time': '',
                    #     'school': '',
                    #     'college': '',
                    #     'pro': '',
                    #     'degree': '',
                    #     'edu': ''
                    # },
                ],

                'current': [
                    # {
                    #     'time': '',
                    #     'school': '',
                    #     'college': '',
                    #     'pro': '',
                    #     'degree': '',
                    #     'edu': ''
                    # },
                ]
            },

            'work_exp': {
                'past': [
                    # {
                    #     'time': '',
                    #     'place': '',
                    #     'job': []
                    # },
                ],

                'current': [
                    # {
                    #     'time': '',
                    #     'place': '',
                    #     'job': []
                    # },
                ]
            }
        }

        # 学习经历
        sexp = {
            'time': '',
            'school': '',
            # 'college': '',
            'pro': '',
            'degree': '',
            'edu': ''
        }

        # 工作经历
        wexp = {
            'time': '',
            'place': '',
            'department': '',
            'job': ''
        }

        # 复制sexp
        def sexp_copy():
            new_sexp = sexp.copy()
            return new_sexp

        # 复制wexp
        def wexp_copy():
            new_wexp = wexp.copy()
            # new_wexp['job'] = wexp['job'].copy()
            return new_wexp

        # 判断sexp字典是否含有信息
        def sexp_contain_msg():
            if sexp['time'] != '' or sexp['school'] != '' or sexp['pro'] != '' or \
                    sexp['degree'] != '' or sexp['edu'] != '':
                return True
            return False

        # 判断wexp字典是否含有信息
        def wexp_contain_msg():
            if wexp['time'] != '' or wexp['place'] != '' or wexp['department'] != '' or wexp['job'] != '':
                return True
            return False

        # 把sexp字典内的信息清空
        def clear_sexp():
            sexp['time'] = ''
            sexp['school'] = ''
            # sexp['college'] = ''
            sexp['pro'] = ''
            sexp['degree'] = ''
            sexp['edu'] = ''

        # 把wexp字典内的信息清空
        def clear_wexp():
            wexp['time'] = ''
            wexp['place'] = ''
            wexp['department'] = ''
            wexp['job'] = ''

        def before_words_deal():
            pass

        previous_time_label = ''  # 保存上一个时间标签 '' ctime ptime
        previous_time = ''  # 保存上一个时间 '' 具体时间
        previous_school = ''  # 保存上一个学校 '' 具体学校
        previous_company = ''  # 保存上一个公司 '' 具体公司
        # previous_department = ''  # 保存上一个部门 '' 具体部门

        # 将实体词处理成格式化信息
        for sentence_label, notional_words, notional_labels in zip(sentence_label_list, notional_words_list,
                                                                   notional_labels_list):
            if sentence_label == 'basic':

                if sexp_contain_msg():
                    if sexp['school'] != '' and sexp['school'] == wexp['place']:  # 要判断是否是在学校工作
                        if wexp['job'] != '':  # 工作不为空 是在学校工作
                            clear_sexp()
                        else:  # 不是在学校工作
                            sexp['time'] = previous_time
                            if sexp['school'] == '':
                                sexp['school'] = previous_school
                            if previous_time_label == 'ctime':
                                regresume['school_exp']['current'].append(sexp_copy())
                            else:
                                regresume['school_exp']['past'].append(sexp_copy())
                            clear_sexp()
                            clear_wexp()
                    else:
                        sexp['time'] = previous_time
                        if sexp['school'] == '':
                            sexp['school'] = previous_school
                        if previous_time_label == 'ctime':
                            regresume['school_exp']['current'].append(sexp_copy())
                        else:
                            regresume['school_exp']['past'].append(sexp_copy())
                        clear_sexp()

                if wexp_contain_msg():  # 判断不是学校
                    wexp['time'] = previous_time
                    if wexp['place'] == '':
                        wexp['place'] = previous_company
                    if previous_time_label == 'ctime':
                        regresume['work_exp']['current'].append(wexp_copy())
                    else:
                        regresume['work_exp']['past'].append(wexp_copy())
                    clear_wexp()

                previous_time_label = ''
                previous_time = ''
                previous_school = ''  # 保存上一个学校 '' 具体学校
                previous_company = ''  # 保存上一个公司 '' 具体公司
                # previous_department = ''  # 保存上一个部门 '' 具体部门

                for notional_word, notional_label in zip(notional_words, notional_labels):
                    if notional_label == 'name':
                        if regresume['basic_info']['name'] == '':
                            regresume['basic_info']['name'] = notional_word
                    elif notional_label == 'sex':
                        if regresume['basic_info']['sex'] == '':
                            if notional_word == '男' or notional_word == '先生':
                                regresume['basic_info']['sex'] = '男'
                            elif notional_word == '女' or notional_word == '女士':
                                regresume['basic_info']['sex'] = '女'
                    elif notional_label == 'nationality':
                        if regresume['basic_info']['nationality'] == '':
                            regresume['basic_info']['nationality'] = notional_word
                    elif notional_label == 'nation':
                        if regresume['basic_info']['nation'] == '':
                            regresume['basic_info']['nation'] = notional_word
                    elif notional_label == 'time':
                        if regresume['basic_info']['born'] == '':
                            regresume['basic_info']['born'] = notional_word

            elif sentence_label == 'ctime':

                if sexp_contain_msg():
                    if sexp['school'] != '' and sexp['school'] == wexp['place']:  # 要判断是否是在学校工作
                        if wexp['job'] != '':  # 工作不为空 是在学校工作
                            clear_sexp()
                        else:  # 不是在学校工作
                            sexp['time'] = previous_time
                            if sexp['school'] == '':
                                sexp['school'] = previous_school
                            if previous_time_label == 'ctime':
                                regresume['school_exp']['current'].append(sexp_copy())
                            else:
                                regresume['school_exp']['past'].append(sexp_copy())
                            clear_sexp()
                            clear_wexp()
                    else:
                        sexp['time'] = previous_time
                        if sexp['school'] == '':
                            sexp['school'] = previous_school
                        if previous_time_label == 'ctime':
                            regresume['school_exp']['current'].append(sexp_copy())
                        else:
                            regresume['school_exp']['past'].append(sexp_copy())
                        clear_sexp()

                if wexp_contain_msg():  # 判断不是学校
                    wexp['time'] = previous_time
                    if wexp['place'] == '':
                        wexp['place'] = previous_company
                    if previous_time_label == 'ctime':
                        regresume['work_exp']['current'].append(wexp_copy())
                    else:
                        regresume['work_exp']['past'].append(wexp_copy())
                    clear_wexp()

                previous_time_label = 'ctime'
                previous_time = ''
                previous_school = ''  # 保存上一个学校 '' 具体学校
                previous_company = ''  # 保存上一个公司 '' 具体公司
                # previous_department = ''  # 保存上一个部门 '' 具体部门

                for notional_word, notional_label in zip(notional_words, notional_labels):
                    if notional_label == 'time':
                        if sexp_contain_msg():
                            if sexp['school'] != '' and sexp['school'] == wexp['place']:  # 要判断是否是在学校工作
                                if wexp['job'] != '':  # 工作不为空 是在学校工作
                                    clear_sexp()
                                else:  # 不是在学校工作
                                    sexp['time'] = previous_time
                                    if sexp['school'] == '':
                                        sexp['school'] = previous_school
                                        regresume['school_exp']['current'].append(sexp_copy())
                                    clear_sexp()
                                    clear_wexp()
                            else:
                                sexp['time'] = previous_time
                                if sexp['school'] == '':
                                    sexp['school'] = previous_school
                                    regresume['school_exp']['current'].append(sexp_copy())
                                clear_sexp()

                        if wexp_contain_msg():  # 判断不是学校
                            wexp['time'] = previous_time
                            if wexp['place'] == '':
                                wexp['place'] = previous_company
                            if previous_time_label == 'ctime':
                                regresume['work_exp']['current'].append(wexp_copy())
                            else:
                                regresume['work_exp']['past'].append(wexp_copy())
                            clear_wexp()

                        previous_time = notional_word
                        previous_school = ''  # 保存上一个学校 '' 具体学校
                        previous_company = ''  # 保存上一个公司 '' 具体公司
                        # previous_department = ''  # 保存上一个部门 '' 具体部门

                    elif notional_label == 'school':
                        if sexp['school'] != '':
                            sexp['time'] = previous_time
                            regresume['school_exp']['current'].append(sexp_copy())
                            clear_sexp()
                        previous_school = notional_word
                        sexp['school'] = notional_word

                        if wexp['place'] != '':
                            wexp['time'] = previous_time
                            regresume['work_exp']['current'].append(wexp_copy())
                            clear_wexp()
                        previous_company = notional_word
                        wexp['place'] = notional_word
                        # previous_department = ''

                    # elif notional_label == 'college':
                    #     if sexp['college'] != '':
                    #         sexp['time'] = previous_time
                    #         regresume['school_exp']['current'].append(sexp_copy())
                    #         clear_sexp()
                    #     else:
                    #         sexp['college'] = notional_word

                    elif notional_label == 'pro':
                        if sexp['pro'] != '':
                            sexp['time'] = previous_time
                            if sexp['school'] == '':
                                sexp['school'] = previous_school
                            regresume['school_exp']['current'].append(sexp_copy())
                            clear_sexp()
                            clear_wexp()
                        sexp['pro'] = notional_word

                    elif notional_label == 'degree':
                        if sexp['degree'] != '':
                            sexp['time'] = previous_time
                            if sexp['school'] == '':
                                sexp['school'] = previous_school
                            regresume['school_exp']['current'].append(sexp_copy())
                            clear_sexp()
                            clear_wexp()
                        sexp['degree'] = notional_word

                    elif notional_label == 'edu':
                        if sexp['edu'] != '':
                            sexp['time'] = previous_time
                            if sexp['school'] == '':
                                sexp['school'] = previous_school
                            regresume['school_exp']['current'].append(sexp_copy())
                            clear_sexp()
                            clear_wexp()
                        sexp['edu'] = notional_word

                    elif notional_label == 'company':
                        if wexp['place'] != '':
                            wexp['time'] = previous_time
                            regresume['work_exp']['current'].append(wexp_copy())
                            clear_wexp()
                        previous_company = notional_word
                        wexp['place'] = notional_word
                        # previous_department = ''

                    elif notional_label == 'department':
                        if wexp['job'] != '':
                            wexp['time'] = previous_time
                            if wexp['place'] == '':
                                wexp['place'] = previous_company
                            # if wexp['department'] == '':
                            #     wexp['department'] = previous_department
                            regresume['work_exp']['current'].append(wexp_copy())
                            clear_wexp()

                        if wexp['department'] != '':
                            wexp['time'] = previous_time
                            if wexp['place'] == '':
                                wexp['place'] = previous_company
                            regresume['work_exp']['current'].append(wexp_copy())
                            clear_wexp()
                        # previous_department = notional_word
                        wexp['department'] = notional_word

                    elif notional_label == 'job':
                        if wexp['job'] != '':
                            wexp['time'] = previous_time
                            if wexp['place'] == '':
                                wexp['place'] = previous_company
                            # if wexp['department'] == '':
                            #     wexp['department'] = previous_department
                            regresume['work_exp']['current'].append(wexp_copy())
                            clear_wexp()
                        wexp['job'] = notional_word
                        # print('test', 'wexp', previous_company)

            elif sentence_label == 'ptime':

                if sexp_contain_msg():
                    if sexp['school'] != '' and sexp['school'] == wexp['place']:  # 要判断是否是在学校工作
                        if wexp['job'] != '':  # 工作不为空 是在学校工作
                            clear_sexp()
                        else:  # 不是在学校工作
                            sexp['time'] = previous_time
                            if sexp['school'] == '':
                                sexp['school'] = previous_school
                            if previous_time_label == 'ctime':
                                regresume['school_exp']['current'].append(sexp_copy())
                            else:
                                regresume['school_exp']['past'].append(sexp_copy())
                            clear_sexp()
                            clear_wexp()
                    else:
                        sexp['time'] = previous_time
                        if sexp['school'] == '':
                            sexp['school'] = previous_school
                        if previous_time_label == 'ctime':
                            regresume['school_exp']['current'].append(sexp_copy())
                        else:
                            regresume['school_exp']['past'].append(sexp_copy())
                        clear_sexp()

                if wexp_contain_msg():  # 判断不是学校
                    wexp['time'] = previous_time
                    if wexp['place'] == '':
                        wexp['place'] = previous_company
                    if previous_time_label == 'ctime':
                        regresume['work_exp']['current'].append(wexp_copy())
                    else:
                        regresume['work_exp']['past'].append(wexp_copy())
                    clear_wexp()

                previous_time_label = 'ptime'
                previous_time = ''
                previous_school = ''  # 保存上一个学校 '' 具体学校
                previous_company = ''  # 保存上一个公司 '' 具体公司
                # previous_department = ''  # 保存上一个部门 '' 具体部门

                for notional_word, notional_label in zip(notional_words, notional_labels):
                    if notional_label == 'time':
                        if sexp_contain_msg():
                            if sexp['school'] != '' and sexp['school'] == wexp['place']:  # 要判断是否是在学校工作
                                if wexp['job'] != '':  # 工作不为空 是在学校工作
                                    clear_sexp()
                                else:  # 不是在学校工作
                                    sexp['time'] = previous_time
                                    if sexp['school'] == '':
                                        sexp['school'] = previous_school
                                        regresume['school_exp']['past'].append(sexp_copy())
                                    clear_sexp()
                                    clear_wexp()
                            else:
                                sexp['time'] = previous_time
                                if sexp['school'] == '':
                                    sexp['school'] = previous_school
                                    regresume['school_exp']['past'].append(sexp_copy())
                                clear_sexp()

                        if wexp_contain_msg():  # 判断不是学校
                            wexp['time'] = previous_time
                            if wexp['place'] == '':
                                wexp['place'] = previous_company
                            if previous_time_label == 'ctime':
                                regresume['work_exp']['current'].append(wexp_copy())
                            else:
                                regresume['work_exp']['past'].append(wexp_copy())
                            clear_wexp()

                        previous_time = notional_word
                        previous_school = ''  # 保存上一个学校 '' 具体学校
                        previous_company = ''  # 保存上一个公司 '' 具体公司
                        # previous_department = ''  # 保存上一个部门 '' 具体部门

                    elif notional_label == 'school':
                        if sexp['school'] != '':
                            sexp['time'] = previous_time
                            regresume['school_exp']['past'].append(sexp_copy())
                            clear_sexp()
                        previous_school = notional_word
                        sexp['school'] = previous_school

                        if wexp['place'] != '':
                            wexp['time'] = previous_time
                            regresume['work_exp']['past'].append(wexp_copy())
                            clear_wexp()
                        previous_company = notional_word
                        wexp['place'] = previous_company
                        # previous_department = ''

                    # elif notional_label == 'college':
                    #     if sexp['college'] != '':
                    #         sexp['time'] = previous_time
                    #         regresume['school_exp']['past'].append(sexp_copy())
                    #         clear_sexp()
                    #     else:
                    #         sexp['college'] = notional_word

                    elif notional_label == 'pro':
                        if sexp['pro'] != '':
                            sexp['time'] = previous_time
                            if sexp['school'] == '':
                                sexp['school'] = previous_school
                            regresume['school_exp']['past'].append(sexp_copy())
                            clear_sexp()
                            clear_wexp()
                        sexp['pro'] = notional_word

                    elif notional_label == 'degree':
                        if sexp['degree'] != '':
                            sexp['time'] = previous_time
                            if sexp['school'] == '':
                                sexp['school'] = previous_school
                            regresume['school_exp']['past'].append(sexp_copy())
                            clear_sexp()
                            clear_wexp()
                        sexp['degree'] = notional_word

                    elif notional_label == 'edu':
                        if sexp['edu'] != '':
                            sexp['time'] = previous_time
                            if sexp['school'] == '':
                                sexp['school'] = previous_school
                            regresume['school_exp']['past'].append(sexp_copy())
                            clear_sexp()
                            clear_wexp()
                        sexp['edu'] = notional_word

                    elif notional_label == 'company':
                        if wexp['place'] != '':
                            wexp['time'] = previous_time
                            regresume['work_exp']['past'].append(wexp_copy())
                            clear_wexp()
                        previous_company = notional_word
                        wexp['place'] = notional_word
                        # previous_department = ''

                    elif notional_label == 'department':
                        if wexp['job'] != '':
                            wexp['time'] = previous_time
                            if wexp['place'] == '':
                                wexp['place'] = previous_company
                            # if wexp['department'] == '':
                            #     wexp['department'] = previous_department
                            regresume['work_exp']['past'].append(wexp_copy())
                            clear_wexp()

                        if wexp['department'] != '':
                            wexp['time'] = previous_time
                            if wexp['place'] == '':
                                wexp['place'] = previous_company
                            regresume['work_exp']['past'].append(wexp_copy())
                            clear_wexp()
                        # previous_department = notional_word
                        wexp['department'] = notional_word

                    elif notional_label == 'job':
                        if wexp['job'] != '':
                            wexp['time'] = previous_time
                            if wexp['place'] == '':
                                wexp['place'] = previous_company
                            # if wexp['department'] == '':
                            #     wexp['department'] = previous_department
                            regresume['work_exp']['past'].append(wexp_copy())
                            clear_wexp()
                        wexp['job'] = notional_word
                        # print('test', 'wexp', previous_company)

            elif sentence_label == 'sexp':
                for notional_word, notional_label in zip(notional_words, notional_labels):
                    if notional_label == 'school':
                        if sexp['school'] != '':
                            sexp['time'] = previous_time
                            if previous_time_label == 'ctime':
                                regresume['school_exp']['current'].append(sexp_copy())
                            else:
                                regresume['school_exp']['past'].append(sexp_copy())
                            clear_sexp()
                        previous_school = notional_word
                        sexp['school'] = notional_word

                    # elif notional_label == 'college':
                    #     if sexp['college'] != '':
                    #         sexp['time'] = previous_time
                    #         if previous_time_label == 'ctime':
                    #             regresume['school_exp']['current'].append(sexp_copy())
                    #         else:
                    #             regresume['school_exp']['past'].append(sexp_copy())
                    #         clear_sexp()
                    #     else:
                    #         sexp['college'] = notional_word

                    elif notional_label == 'pro':
                        if sexp['pro'] != '':
                            sexp['time'] = previous_time
                            if sexp['school'] == '':
                                sexp['school'] = previous_school
                            if previous_time_label == 'ctime':
                                regresume['school_exp']['current'].append(sexp_copy())
                            else:
                                regresume['school_exp']['past'].append(sexp_copy())
                            clear_sexp()
                            clear_wexp()
                        sexp['pro'] = notional_word

                    elif notional_label == 'degree':
                        if sexp['degree'] != '':
                            sexp['time'] = previous_time
                            if sexp['school'] == '':
                                sexp['school'] = previous_school
                            if previous_time_label == 'ctime':
                                regresume['school_exp']['current'].append(sexp_copy())
                            else:
                                regresume['school_exp']['past'].append(sexp_copy())
                            clear_sexp()
                            clear_wexp()
                        sexp['degree'] = notional_word

                    elif notional_label == 'edu':
                        if sexp['edu'] != '':
                            sexp['time'] = previous_time
                            if sexp['school'] == '':
                                sexp['school'] = previous_school
                            if previous_time_label == 'ctime':
                                regresume['school_exp']['current'].append(sexp_copy())
                            else:
                                regresume['school_exp']['past'].append(sexp_copy())
                            clear_sexp()
                            clear_wexp()
                        sexp['edu'] = notional_word

            elif sentence_label == 'wexp':
                for notional_word, notional_label in zip(notional_words, notional_labels):
                    if notional_label == 'company':
                        if wexp['place'] != '':
                            wexp['time'] = previous_time
                            if previous_time_label == 'ctime':
                                regresume['work_exp']['current'].append(wexp_copy())
                            else:
                                regresume['work_exp']['past'].append(wexp_copy())
                            clear_wexp()
                        previous_company = notional_word
                        wexp['place'] = notional_word
                        # previous_department = ''

                    elif notional_label == 'department':
                        if wexp['job'] != '':
                            wexp['time'] = previous_time
                            if wexp['place'] == '':
                                wexp['place'] = previous_company
                            # if wexp['department'] == '':
                            #     wexp['department'] = previous_department
                            if previous_time_label == 'ctime':
                                regresume['work_exp']['current'].append(wexp_copy())
                            else:
                                regresume['work_exp']['past'].append(wexp_copy())
                            clear_wexp()

                        if wexp['department'] != '':
                            wexp['time'] = previous_time
                            if wexp['place'] == '':
                                wexp['place'] = previous_company
                            if previous_time_label == 'ctime':
                                regresume['work_exp']['current'].append(wexp_copy())
                            else:
                                regresume['work_exp']['past'].append(wexp_copy())
                            clear_wexp()
                        # previous_department = notional_word
                        wexp['department'] = notional_word

                    elif notional_label == 'job':
                        if wexp['job'] != '':
                            wexp['time'] = previous_time
                            if wexp['place'] == '':
                                wexp['place'] = previous_company
                            # if wexp['department'] == '':
                            #     wexp['department'] = previous_department
                            if previous_time_label == 'ctime':
                                regresume['work_exp']['current'].append(wexp_copy())
                            else:
                                regresume['work_exp']['past'].append(wexp_copy())
                            clear_wexp()
                        wexp['job'] = notional_word
                        # print('test', 'wexp', previous_company)

            elif sentence_label == 'noinfo':

                if sexp_contain_msg():
                    if sexp['school'] != '' and sexp['school'] == wexp['place']:  # 要判断是否是在学校工作
                        if wexp['job'] != '':  # 工作不为空 是在学校工作
                            clear_sexp()
                        else:  # 不是在学校工作
                            sexp['time'] = previous_time
                            if sexp['school'] == '':
                                sexp['school'] = previous_school
                            if previous_time_label == 'ctime':
                                regresume['school_exp']['current'].append(sexp_copy())
                            else:
                                regresume['school_exp']['past'].append(sexp_copy())
                            clear_sexp()
                            clear_wexp()
                    else:
                        sexp['time'] = previous_time
                        if sexp['school'] == '':
                            sexp['school'] = previous_school
                        if previous_time_label == 'ctime':
                            regresume['school_exp']['current'].append(sexp_copy())
                        else:
                            regresume['school_exp']['past'].append(sexp_copy())
                        clear_sexp()

                if wexp_contain_msg():  # 判断不是学校
                    wexp['time'] = previous_time
                    if wexp['place'] == '':
                        wexp['place'] = previous_company
                    if previous_time_label == 'ctime':
                        regresume['work_exp']['current'].append(wexp_copy())
                    else:
                        regresume['work_exp']['past'].append(wexp_copy())
                    clear_wexp()

                previous_time_label = ''
                previous_time = ''
                previous_school = ''  # 保存上一个学校 '' 具体学校
                previous_company = ''  # 保存上一个公司 '' 具体公司
                # previous_department = ''  # 保存上一个部门 '' 具体部门

        # 结束判断sexp、wexp内还有数据
        if sexp_contain_msg():
            if sexp['school'] != '' and sexp['school'] == wexp['place']:  # 要判断是否是在学校工作
                if wexp['job'] != '':  # 工作不为空 是在学校工作
                    clear_sexp()
                else:  # 不是在学校工作
                    sexp['time'] = previous_time
                    if sexp['school'] == '':
                        sexp['school'] = previous_school
                    if previous_time_label == 'ctime':
                        regresume['school_exp']['current'].append(sexp_copy())
                    else:
                        regresume['school_exp']['past'].append(sexp_copy())
                    clear_sexp()
                    clear_wexp()
            else:
                sexp['time'] = previous_time
                if sexp['school'] == '':
                    sexp['school'] = previous_school
                if previous_time_label == 'ctime':
                    regresume['school_exp']['current'].append(sexp_copy())
                else:
                    regresume['school_exp']['past'].append(sexp_copy())
                clear_sexp()

        if wexp_contain_msg():  # 判断不是学校
            wexp['time'] = previous_time
            if wexp['place'] == '':
                wexp['place'] = previous_company
            if previous_time_label == 'ctime':
                regresume['work_exp']['current'].append(wexp_copy())
            else:
                regresume['work_exp']['past'].append(wexp_copy())
            clear_wexp()

        return regresume


if __name__ == '__main__':
    resume = "sdfsdfsdfsdf,sfsdfsdfsdf"
    extractor = Extractor()
    extractor.single_extract(resume)
    extractor.batch_extract([resume])
