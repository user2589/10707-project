#!/usr/bin/env python3
# coding=utf-8

import numpy as np
import pandas as pd


def get_train_data():
    return pd.read_csv("input/test_stage_1.tsv", sep="\t", index_col=0)


def get_train_labels():
    return pd.read_csv('input/right_answers.csv', index_col=0).values


def score(labels):
    """
    Arguments:
        labels: one-hot encoded nx3 array of labels [A, B, NEITHER]
    """
    a_bit = 10e-15
    right_answers = get_train_labels()
    l = np.maximum(np.minimum(labels, 1-a_bit), a_bit)
    # normalize rows to sum up to 1
    l = l / np.tile(l.sum(axis=1), (3, 1)).T
    return -(right_answers * np.log(l)).sum() / len(l)


def accuracy(labels):
    right_answers = get_train_labels()
    n = len(labels)
    one_hot = np.zeros(right_answers.shape)
    one_hot[np.arange(n), np.argmax(labels)] = 1
    return (one_hot * right_answers).sum() / n
