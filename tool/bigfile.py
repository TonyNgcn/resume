#!/usr/bin/python 
# -*- coding: utf-8 -*-
import logging


def get_lines(filepath: str):
    try:
        file = open(filepath, "r")
        for line in file:
            yield line
        file.close()

    except Exception as e:
        logging.error(e)
        exit(1)
