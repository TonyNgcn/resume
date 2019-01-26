#!/usr/bin/python 
# -*- coding: utf-8 -*-

from sqlalchemy import Column, Integer, VARCHAR, TEXT
from sqlalchemy.ext.declarative import declarative_base

# 生成orm基类
Base = declarative_base()


class Label(Base):
    __tablename__ = "label"
    id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(VARCHAR(20), nullable=False)


class Mark(Base):
    __tablename__ = "mark"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(TEXT, nullable=False)
    label_mark = Column(VARCHAR(20), nullable=False)
    num = Column(Integer, nullable=False, default=0)


class Data(Base):
    __tablename__ = "data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(TEXT, nullable=False)
    label = Column(VARCHAR(20), nullable=False)
