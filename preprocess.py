# tokenize data
# partition data
# generate vocabulary
import numpy as np
import os
import sys
import random
from collections import Counter
import nltk
from rakutenma import RakutenMA
import config

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

def english_tokenizer(text_list):
    return [nltk.word_tokenize(text) for text in text_list]

def write_data_to_file(data, file_path):
    text_list = [' '.join(tokenized_text) for tokenized_text in data]
    with open(file_path, 'w') as fout:
        for text in text_list:
            fout.write(text + "\n")

def generate_vocabulary(data, threshold):
    counter = Counter()
    for tokenized_text in data:
        counter.update(tokenized_text)
    for word, num in list(counter.items()):
        if num < threshold:
            del counter[word]
    return list(counter.keys())

def make_vocab_file(file_path, vocab):
    with open(file_path, 'w') as fout:
        for word in vocab:
            fout.write(word + "\n")

def read_file(file_path):
    with open(file_path) as fin:
        return fin.read().splitlines()
    
if __name__ == "__main__":
    assert len(sys.argv) == 2
    data_path = sys.argv[1] 
    param_dict = {"iknow" : config.IKNOW_DATA_PARAMS, "jesc" : config.JESC_DATA_PARAMS}
    data_params = param_dict[data_path]

    japanese_tokenizer = Japanese_Tokenizer()
#    print("Tokenizing text...")
#    with open(os.path.join(data_path, "en")) as fin:
#        text_list = fin.read().splitlines()
#        tokenized_en = english_tokenizer(text_list)
#    with open(os.path.join(data_path, "ja")) as fin:
#        text_list = fin.read().splitlines()
#        tokenized_ja = japanese_tokenizer(text_list)
#
#    print("Partitioning data...")
#    en_ja_pairs = list(zip(tokenized_en, tokenized_ja))
#    random.shuffle(en_ja_pairs)
#    validate_size = data_params["validate_size"]
#    train_data = en_ja_pairs[: -validate_size]
#    validate_data = en_ja_pairs[-validate_size :]
#    train_en, train_ja = zip(*train_data)
#    validate_en, validate_ja = zip(*validate_data)
#    write_data_to_file(train_en, os.path.join(data_path, "train_en"))
#    write_data_to_file(train_ja, os.path.join(data_path, "train_ja"))
#    write_data_to_file(validate_en, os.path.join(data_path, "validate_en"))
#    write_data_to_file(validate_ja, os.path.join(data_path, "validate_ja"))

    print("Generating vocabulary...")

    train_en = english_tokenizer(read_file(os.path.join(data_path, "train_en")))
    train_ja = japanese_tokenizer(read_file(os.path.join(data_path, "train_ja")))
    vocab_en = generate_vocabulary(train_en, data_params["vocab_en_threshold"])
    vocab_ja = generate_vocabulary(train_ja, data_params["vocab_ja_threshold"])
    symbols = ["<PAD>", "<GO>", "<EOS>", "<UNK>"]
    make_vocab_file(os.path.join(data_path, "vocab_en"), symbols + vocab_en)
    make_vocab_file(os.path.join(data_path, "vocab_ja"), symbols + vocab_ja)
    print("English vocabulary size: {}".format(len(vocab_en)))
    print("Japanese vocabulary size: {}".format(len(vocab_ja)))
