import tensorflow as tf
import math
import numpy as np
from collections import defaultdict
import nltk
from rakutenma import RakutenMA
from nltk.tokenize.treebank import TreebankWordDetokenizer

class Japanese_Tokenizer:
    def __init__(self):
        rma = RakutenMA()
        rma.load("model_ja.json")
        rma.hash_func = rma.create_hash_func(15)
        self.rma = rma 

    def __call__(self, text_list):
        return [self.tokenize(text) for text in text_list]
        
    def tokenize(self, text):
        x = self.rma.tokenize(text)
        return [t[0] for t in x]

class English_Tokenizer:
    def __init__(self):
        pass

    def __call__(self, text_list):
        return [self.tokenize(text) for text in text_list]

    def tokenize(self, text):
        return nltk.word_tokenize(text)

class Japanese_Detokenizer:
    def __init__(self):
        pass

    def __call__(self, tokens_list):
        return [self.detokenize(tokens) for tokens in tokens_list]

    def detokenize(self, tokens):
        return ''.join(tokens)

class English_Detokenizer:
    def __init__(self):
        self.twd = TreebankWordDetokenizer()

    def __call__(self, tokens_list):
        return [self.detokenize(tokens) for tokens in tokens_list]

    def detokenize(self, tokens):
        return self.twd.detokenize(tokens)

class Word_Id_Converter:
    def __init__(self, vocab_file_path, default_id, default_word="<DEFAULT>"):
        with open(vocab_file_path) as fin:
            words = fin.read().splitlines()
        self.word2id_table = self.get_lookup_table(words, list(range(len(words))), default_id)
        self.id2word_table = self.get_lookup_table(list(range(len(words))), words, default_word)

    def word2id(self, words_tensor):
        return self.word2id_table.lookup(words_tensor)

    def id2word(self, ids_tensor):
        return self.id2word_table.lookup(ids_tensor)

    def get_initializer(self, keys, values):
        return tf.contrib.lookup.KeyValueTensorInitializer(keys, values)

    def get_lookup_table(self, keys, values, default_value):
        initializer = self.get_initializer(keys, values)
        return tf.contrib.lookup.HashTable(initializer, default_value)

def get_padded_positions(inputs, padding_value=0):
    return tf.to_float(tf.equal(inputs, padding_value))

def get_file_size(file_path):
    with open(file_path, 'r') as f:
        return len(list(f))

def get_source_data(data_file_path, converter, unk_id):
    dataset = tf.data.TextLineDataset(data_file_path)
    dataset = dataset.map(lambda text : tf.string_split([text]).values)
    dataset = dataset.map(lambda tokenized_text : converter.word2id(tokenized_text))
    return dataset

def get_target_data(data_file_path, converter, go_id, eos_id, unk_id):
    dataset = get_source_data(data_file_path, converter, unk_id)
    target_inputs = dataset.map(lambda id_seq: tf.concat([tf.constant([go_id], tf.int32), id_seq], axis=0))
    target_outputs = dataset.map(lambda id_seq: tf.concat([id_seq, tf.constant([eos_id], tf.int32)], axis=0))
    return target_inputs, target_outputs

def prepare_dataset(source_data_file, target_data_file, source_converter, target_converter, go_id, eos_id, unk_id):
    source_inputs = get_source_data(source_data_file, source_converter, unk_id)
    target_inputs, target_outputs = get_target_data(target_data_file, target_converter, go_id, eos_id, unk_id)
    dataset = tf.data.Dataset.zip((source_inputs, target_inputs, target_outputs))
    return dataset

def get_tokenizer(lang, bpe=False):
    if not bpe:
        if lang in ["en"]:
            return nltk.word_tokenize
        if lang in ["ja"]:
            rma = RakutenMA()
            rma.load('model_ja.jason')
            rma.hash_func = rma.create_hash_func(15)
            return lambda x : [token[0] for token in rma.tokenize(x)]

def get_detokenizer(lang, bpe=False):
    if not bpe:
        if lang in ["en"]:
            detokenizer = TreebankWordDetokenizer()
            return lambda x : detokenizer.detokenize(x)
        if lang in ["ja"]:
            return lambda x : ''.join(x)

def dynamic_lstm(cell, inputs, output_mask=None, initial_state=None):
# inputs shape: [batch_size, max_len, embedding_dim]
    batch_size, max_len, _ = inputs.shape
    c_size, h_size = cell._state_size
    output_0 = tf.zeros([batch_size, 1, h_size])
    if initial_state is None:
        c0 = tf.zeros([batch_size, c_size])
        h0 = tf.zeros([batch_size, h_size])
    else:
        c0, h0 = initial_state
    i0 = tf.constant(0)
    def step(outputs, c, h, i):
        _output, _state = cell(inputs[:, i, :], (c, h))
        outputs = tf.concat((outputs, tf.expand_dims(_output, axis=1)), axis=1)
        c, h = _state 
        i += 1
        return outputs, c, h, i
    outputs, c, h, _ = tf.while_loop(lambda output_0, c0, h0, i0 : i0 < max_len, step, [output_0, c0, h0, i0], shape_invariants=[tf.TensorShape([batch_size , None, h_size]), c0.shape, h0.shape, i0.shape]) 
    if output_mask is not None:
        outputs = outputs[:, 1:, :] * output_mask
    return outputs, (c, h)
