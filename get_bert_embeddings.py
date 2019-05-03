#!/usr/bin/env python3
# coding=utf-8

import argparse
import logging
import os

import numpy as np
import pandas as pd

from extract_features import *


def bert_embeddings(
        sequences,
        batch_size=8, max_seq_length=256, layer_indexes=(-1, -2, -3, -4),
        model='uncased_L-12_H-768_A-12',
        do_lower_case=True, use_one_hot_embeddings=False,
        use_tpu=False, num_tpu_cores=8, master=None):
    """

    Args:
        sequences: an iterable of strings. Each string represents a sequence
        batch_size:
        max_seq_length:
        layer_indexes: which layer weights to return
        model:
        do_lower_case:
        use_one_hot_embeddings:
        use_tpu: self explanatory
        num_tpu_cores: self explanatory
        master: If using a TPU, the address of the master

    Returns:
        (Generator[Dict]): results dictionary with fields: {
                'linex_index': sample unique id (actually, just int)
                'features': list of features
                    list is made out of tokens. It starts with [CLS] and ends with [SEP]
                    Each list element is a dict with two keys:
                        'token': well, the token
                        'layers': list of layers

                        Each layer is a dict with two keys:
                            'index': index (-1, -2 etc)
                            'values': list of values, length is up to the model.
            }
    """
    models_dir = 'bert_models'
    assert model in ('uncased_L-12_H-768_A-12', 'cased_L-12_H-768_A-12',
                     'uncased_L-24_H-1024_A-16'), "Unknown BERT model"
    model_path = os.path.join(models_dir, model)
    if not os.path.isdir(model_path):
        output_fname = model + ".zip"
        output_path = os.path.join(models_dir, output_fname)
        logging.warning('Getting the model (will take few minutes).....')
        if os.system('wget https://storage.googleapis.com/bert_models/'
                     '2018_10_18/%s -O %s' % (output_fname, output_path)):
            print('failed to download model')
            exit(1)
        logging.warning('.. done, extracting ..')
        if os.system('unzip %s -d %s' % (output_path, models_dir)):
            print('failed to extract model')
            exit(1)
        logging.warning('.. done, continuing with the model')

    vocab_file = os.path.join(model_path, "vocab.txt")
    init_checkpoint = os.path.join(model_path, "bert_model.ckpt")
    bert_config_file = os.path.join(model_path, "bert_config.json")

    # tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    tpu_config = tf.contrib.tpu.TPUConfig(
        num_shards=num_tpu_cores, per_host_input_for_training=is_per_host)
    run_config = tf.contrib.tpu.RunConfig(master=master, tpu_config=tpu_config)

    examples = [InputExample(i, sentence, None)
                for i, sentence in enumerate(sequences)]

    features = convert_examples_to_features(
        examples=examples, seq_length=max_seq_length, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=init_checkpoint,
        layer_indexes=layer_indexes,
        use_tpu=use_tpu,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=use_tpu,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=batch_size)

    input_fn = input_fn_builder(
        features=features, seq_length=max_seq_length)

    for result in estimator.predict(input_fn, yield_single_examples=True):
        unique_id = int(result["unique_id"])
        feature = unique_id_to_feature[unique_id]
        embeddings = collections.OrderedDict()
        embeddings["linex_index"] = unique_id
        all_features = []
        for (i, token) in enumerate(feature.tokens):
            all_layers = []
            for (j, layer_index) in enumerate(layer_indexes):
                layer_output = result["layer_output_%d" % j]
                layers = collections.OrderedDict()
                layers["index"] = layer_index
                layers["values"] = [
                    round(float(x), 6) for x in layer_output[i:(i + 1)].flat
                ]
                all_layers.append(layers)
            features = collections.OrderedDict()
            features["token"] = token
            features["layers"] = all_layers
            all_features.append(features)
        embeddings["features"] = all_features
        yield embeddings


def get_token_index(text, offset, tokens):
    """
    >>> text = "A quick brownDasheyBhgx fox jump's over the lazy dog"
    >>> tokens = ['[CLS]', 'a', 'quick', 'brown', '##das', '##hey', '##bh',
    ...           '##g', '##x', 'fox', 'jump', "'", 's', 'over', 'the', 'lazy',
    ...           'dog', '[SEP]']
    >>> tokens[get_token_index(text, text.index('fox'), tokens)] == 'fox'
    True
    >>> tokens[get_token_index(text, text.index('over'), tokens)] == 'over'
    True
    """
    # the idea is, match tokens chunk by chunk
    i = 1  # skip [CLS] in the beginning
    for chunk in text[:offset].split():
        acc = ''
        while len(acc) < len(chunk):
            acc += tokens[i].lstrip('##')
            i += 1
            if i >=len(tokens):
                break
    return i


def commalist(l):
    return ','.join(str(el) for el in l)


def get_all_embeddings(input_df):
    """
    Version of transform that returns full list of embeddings

    Returns:
        pd.Dataframe: columns:
            'tokens': list of tokens (strings)
            'embeddings': len(tokens) x 768 numpy array of floats
            'Pronoun-offset': index of the pronoun in the tokens
            'A-offsets': list of A offsets in the tokens
            'B-offsets': list of B offsets in the tokens
    """
    df = input_df.reset_index()
    sequences = df['Text']
    output = pd.DataFrame(
        None, index=df.index,
        columns=('tokens', 'embeddings',
                 'Pronoun-offset', 'A-offsets', 'B-offsets'))

    for embedding in bert_embeddings(
            sequences, layer_indexes=(-1,), max_seq_length=384):
        idx = embedding['linex_index']
        sequence = sequences[idx]
        features = embedding['features']
        tokens = [t['token'] for t in features]
        pronoun_idx = get_token_index(
            sequence, df.loc[idx, 'Pronoun-offset'], tokens)
        a = df.loc[idx, 'A'].lower()
        b = df.loc[idx, 'B'].lower()
        a_indexes = [i for i, token in enumerate(tokens) if token == a]
        b_indexes = [i for i, token in enumerate(tokens) if token == b]
        embeddings = np.array([t['layers'][0]['values'] for t in features])

        output.loc[idx] = {
            'tokens': tokens,
            'embeddings': embeddings,
            'Pronoun-offset': pronoun_idx,
            'A-offsets': a_indexes,
            'B-offsets': b_indexes
        }

    return output


def transform(input_df):
    df = input_df.reset_index()
    sequences = df['Text']
    output = pd.DataFrame(None, index=df.index,
                          columns=('Pronoun', 'A', 'B', 'A_avg', 'B_avg'))

    # I'm not sure if it is safe to assume embeddings will come in order,
    # so thhere are few steps to make sure
    # Also, note that we need more than 256 tokens
    # because of punctuation and quotes
    for embedding in bert_embeddings(
            sequences, layer_indexes=(-1,), max_seq_length=384):
        idx = embedding['linex_index']
        sequence = sequences[idx]
        features = embedding['features']
        tokens = [t['token'] for t in features]
        try:
            pronoun_idx = get_token_index(
                sequence, df.loc[idx, 'Pronoun-offset'], tokens)
        except:
            print(sequences.loc[idx])
            print(df.loc[idx, 'Pronoun-offset'])
            print(tokens)
            raise

        pronoun_embed = features[pronoun_idx]['layers'][0]['values']

        a = df.loc[idx, 'A'].lower()
        b = df.loc[idx, 'B'].lower()
        try:
            a_indexes = [i for i, token in enumerate(tokens) if token == a]
            closest_a_idx = max(a_indexes, key=lambda x: -abs(x-pronoun_idx))
            avg_a = sum(np.array(features[i]['layers'][0]['values']) for i in a_indexes
                        ) / len(a_indexes)
            a_embed = features[closest_a_idx]['layers'][0]['values']

            b_indexes = [i for i, token in enumerate(tokens) if token == b]
            closest_b_idx = max(b_indexes, key=lambda x: -abs(x-pronoun_idx))
            avg_b = sum(np.array(features[i]['layers'][0]['values']) for i in b_indexes
                        ) / len(b_indexes)
            b_embed = features[closest_b_idx]['layers'][0]['values']
        except:
            print(sequences.loc[idx])
            print(tokens)
            print(a, b)
            raise

        output.loc[idx] = {
            'Pronoun': commalist(pronoun_embed),
            'A': commalist(a_embed),
            'A_avg': commalist(avg_a),
            'B': commalist(b_embed),
            'B_avg': commalist(avg_b)
        }

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=""""
Build BERT embeddings for the pronoun resolution project.

Input file is expected to contain at least three columns:
    'Text': text of the sequence
    'Pronoun-offset': character offset of the pronoun
    'Pronoun': actual pronoun
    'A', 'B': the alternatives
    'A-offset': character offset of the A alternative
    'B-offset': character offset of the B alternative
    
The output is a CSV file with columns:
    'Pronoun': embedding of the pronoun
    'A', 'B': embedding of the closest alternative as a comma separated list
    'A_avg', 'B_avg': averaged embeddings of the alternatives
        e.g. if there are multiple occasions of each in the text
        
    """)
    parser.add_argument('-i', '--input', default="-",
                        type=argparse.FileType('r'),
                        help='Input filename, "-" or skip for stdin')
    parser.add_argument('-o', '--output', default="-",
                        type=argparse.FileType('w'),
                        help='Output filename, "-" or skip for stdout')
    args = parser.parse_args()

    transform(pd.read_csv(args.input)).to_csv(args.output)

