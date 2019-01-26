# 智能简历信息元抽取项目

### 目录介绍
1. cotrain包是存放协同训练代码。
2. extract包是存放抽取简历代码。
3. holder包是存放保存器代码。
4. model包是存放模型代码。
5. preprocess包是存放模型数据预处理代码。
6. tag包是存放使用训练好的模型来标注数据代码。
7. tool包是存放工具代买。

### 文件介绍
1. config.py是配置文件。
2. server.py是启动http服务提供api文件。
3. train_model.py是训练模型文件。
4. train_wordvec.py是训练词向量模型文件。
5. resume_import.py是抽取简历信息元导入到MongoDB数据库文件。