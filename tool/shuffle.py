#!/usr/bin/python 
# -*- coding: utf-8 -*-

import random


def shuffle_both(x: list, y: list):
    both = list(zip(x, y))
    random.shuffle(both)
    shuffle_x, shuffle_y = zip(*both)

    return list(shuffle_x), list(shuffle_y)
