#!/usr/bin/env python3
# coding=utf-8

import argparse
import logging
import os

import numpy as np
import pandas as pd

import string

PRONOUN2GENDER = {
    'he': 'male', 'him': 'male', 'his': 'male',
    'she': 'female', 'her': 'female', 'hers': 'female'}

GENDER2NAME = {
    'female': (
        ('Mary', 'Alice'),
        ('Mary', 'Elizabeth'),
        ('Mary', 'Kate'),
        ('Alice', 'Elizabeth'),
        ('Alice', 'Kate'),
        ('Kate', 'Elizabeth'),
    ),
    'male': (
        ('John', 'Michael'),
        ('John', 'Henry'),
        ('John', 'James'),
        ('Michael', 'Henry'),
        ('Michael', 'James'),
        ('Henry', 'James')
    )
}

    
def replace(text, search, repl, *indexes):
    """
    Replace all occurences of search by repl and update indexes
    """
    search_len = len(search)
    len_diff = search_len - len(repl)
    while True:
        try:
            idx = text.index(search)
        except ValueError:
            return (text, *indexes)

        text = text[:idx] + repl + text[idx+search_len:]
        indexes = [i if i <= idx else i-len_diff for i in indexes]


def transform(row):
    text, po, ao, bo = row['Text'], row['Pronoun-offset'], row['A-offset'], row['B-offset']
    for a_repl, b_repl in GENDER2NAME[PRONOUN2GENDER[row['Pronoun'].lower()]]:
        if a_repl in text or b_repl in text:
            continue
        new_row = row.copy()
        s = sorted((
            (row['A'], 'A', a_repl), (row['B'], 'B', b_repl)),
            key=lambda x: -len(x[0]))
        # sometimes A is part of B or vice versa.
        # So, we need to replace the longest first
        for search, key, repl in s:
            text, po, ao, bo = replace(text, search, repl, po, ao, bo)
            new_row[key] = repl

        for search, repl in (('#', ''), ('`', ''), ('"', ''), ('*', ''),
                             ("--", "-"), (" '", " "), ("' ", " ")):
            text, po, ao, bo = replace(text, search, repl, po, ao, bo)

        new_row['Text'] = text
        new_row['Pronoun-offset'] = po
        new_row['A-offset'] = ao
        new_row['B-offset'] = bo
        return new_row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=""""
    Build BERT embeddings for the pronoun resolution project.

    Input file is expected to contain at least three columns:
        'Text': text of the sequence
        'Pronoun-offset': character offset of the pronoun
        'Pronoun': actual pronoun
        'A', 'B': the alternatives
        'A-offset': character offset of the A alternative
        'B-offset': character offset of the B alternative

    The output is a CSV file with the same columns, 
        but A and B will be replaced with neutral names 
        and all indexes will be updated.
        """)
    parser.add_argument('-i', '--input', default="-",
                        type=argparse.FileType('r'),
                        help='Input filename, "-" or skip for stdin')
    parser.add_argument('-o', '--output', default="-",
                        type=argparse.FileType('w'),
                        help='Output filename, "-" or skip for stdout')
    args = parser.parse_args()

    pd.read_csv(args.input, sep="\t", index_col=0).apply(
        transform, axis=1).to_csv(args.output)

