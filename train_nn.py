#!/usr/bin/env python3
# coding=utf-8

import argparse
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

import utils


def train_nn(embeddings_filename):
    # labels is a df with one-hot encoded answers in order: A, B, Neither
    labels = utils.get_train_labels()
    be = pd.read_csv(embeddings_filename, index_col=0).applymap(
        lambda value: np.array([float(v) for v in value.split(",")]))
    td = np.array(list(be.apply(
        lambda row: np.concatenate(row[['Pronoun', 'A', 'B']]), axis=1)))
    # validation accuracy:
    # single layer, 128/256 units: 80+
    # single layer, 32/64 units: 79
    # no hidden units (softmax layer only): 77..78%
    # with dropout, on 256 units + 0.6 dropout: 80..82 (up to 84.5 once)
    # 128-32-3 with 0.4 do: 80
    model = tf.keras.Sequential([
            # layers.Dropout(0.7),
            layers.Dense(1024, activation='relu', input_shape=(2304,)),
            # layers.Dropout(0.7),
            layers.Dense(3, activation='softmax')
        ])
    model.compile(
        optimizer=tf.train.AdamOptimizer(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    model.fit(td, labels,
              validation_split=0.1,
              batch_size=5, epochs=20)
    # utils.score(model.predict(td))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=""""
Train neural network on top of BERT embeddings for A, B and pronoun.

Input file is expected to contain at least three columns:
    'Pronoun', 'A', 'B': comma-separated list of 786 floats
    
The output file will contain model snapshot.        
    """)
    parser.add_argument('output_filename', type=str,
                        help='File to save trained model to')
    parser.add_argument('embeddings_filename',
                        default="input/test_stage_1_bert_embed.csv",
                        help='Embeddings filename, "-" or skip for stdin')
    args = parser.parse_args()

    model = train_nn(args.input)
    model.save(args.output_filename)
