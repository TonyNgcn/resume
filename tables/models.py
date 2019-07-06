#!/usr/bin/python 
# -*- coding: utf-8 -*-

from sqlalchemy import Column, Integer, VARCHAR, TEXT
from sqlalchemy.ext.declarative import declarative_base

# 生成orm基类
Base = declarative_base()


class SLabel(Base):
    __tablename__ = 'slabel'
    id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(VARCHAR(20), nullable=True)


class SMark(Base):
    __tablename__ = 'smark'
    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(TEXT, nullable=True)
    label_mark = Column(VARCHAR(20), nullable=True)
    num = Column(Integer, default=0, nullable=True)



class SData(Base):
    __tablename__ = "sdata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(TEXT, nullable=False)
    label = Column(VARCHAR(20), nullable=False)

class WLabel(Base):
    __tablename__ = 'wlabel'
    id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(VARCHAR(20), nullable=True)


class WMark(Base):
    __tablename__ = 'wmark'
    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(TEXT, nullable=True)
    label_marks = Column(TEXT, nullable=True)
    num = Column(Integer, default=0, nullable=True)



class WData(Base):
    __tablename__ = "wdata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(TEXT, nullable=False)
    labels = Column(TEXT, nullable=False)
