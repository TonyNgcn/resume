#!/usr/bin/python 
# -*- coding: utf-8 -*-
from pymongo import MongoClient


def connect():
    client = MongoClient("mongodb://127.0.0.1:27017/")
    return client


def get_db(client, database: str):
    db = client[database]
    return db


def get_collection(db, collection: str):
    co = db[collection]
    return co
