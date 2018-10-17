import os
import config
import random
import numpy as np
from argparse import ArgumentParser
import nltk
from rakutenma import RakutenMA
from gensim.models import Word2Vec

def make_dir(path):
	try:
		os.mkdir(path)
	except OSError:
		pass

def _eng_tokenizer(line, rma):
	return nltk.word_tokenize(line)

def _jap_tokenizer(line, rma):
	tokens = rma.tokenize(line)
	return [token[0] for token in tokens]

def _jap_character_tokenizer(line, rma):
	return list(line)

def _get_tokenizer(lang):
	rma = None
	if lang == 'ja':
		rma = RakutenMA()
		rma.load('model_ja.json')
		rma.hash_func = rma.create_hash_func(15)
		tokenizer = _jap_tokenizer
		# tokenizer = _jap_character_tokenizer
	else:
		tokenizer = _eng_tokenizer
	return tokenizer, rma

def tokenize_data(in_name, lang):
	out_name = in_name
	tokenizer, rma = _get_tokenizer(lang)
	in_f = open(os.path.join(config.RAW_DIR, in_name), 'r')
	out_f = open(os.path.join(config.PROC_DIR, out_name), 'w')
	lines = in_f.read().splitlines()
	line_count = 0
	for line in lines:
		# line = line.lower()
		line_count += 1
		if line_count % 1000 == 1:
			print(line_count, 'lines have been processed')
		tokens = tokenizer(line, rma)
		line = ' '.join(tokens)
		out_f.write(line + '\n')
	for f in in_f, out_f:
		f.close()

def build_vocab(lang):
	model = Word2Vec.load(os.path.join(config.PROC_DIR, lang + '.bin'))
	out_name = 'vocab.{}'.format(lang)
	out_f = open(os.path.join(config.PROC_DIR, out_name), 'w')
	config_f = open('config.py', 'a')
	vocab = model.wv.vocab
	# out_f.write('<PAD>' + '\n')
	# out_f.write('<UNK>' + '\n')
	# if phase == 'decode':
	# 	for s in '<GO>', '<EOS>':
	# 		out_f.write(s + '\n')
	# for word in vocab.keys():
	# 	out_f.write(word + '\n')
	# if phase == 'encode':
	# 	vocab_size = 2 + len(vocab)
	# 	config_f.write('ENC_VOCAB_SIZE = {}\n'.format(vocab_size))
	# else:
	# 	vocab_size = 4 + len(vocab)
	# 	config_f.write('DEC_VOCAB_SIZE = {}\n'.format(vocab_size))
	out_f.write('<PAD>' + '\n')
	out_f.write('<UNK>' + '\n')
	out_f.write('<GO>' + '\n')
	out_f.write('<EOS>' + '\n')
	for word in vocab.keys():
		out_f.write(word + '\n')
	vocab_size = 4 + len(vocab)
	config_f.write('{}_VOCAB_SIZE = {}\n'.format(lang.upper(), vocab_size))
	for f in config_f, out_f:
		f.close()

def _get_vocab2id(in_name):
	with open(os.path.join(config.PROC_DIR, in_name), 'r') as in_f:
		vocab = in_f.read().splitlines()
	return {word:i for i, word in enumerate(vocab)}

def add_special_symbols(in_name, use, lang):
	in_f = open(os.path.join(config.PROC_DIR, in_name), 'r')
	out_name = '{}.{}'.format(in_name, use)
	out_f = open(os.path.join(config.PROC_DIR, out_name), 'w')
	vocab2id = _get_vocab2id('vocab.{}'.format(lang))
	lines = in_f.read().splitlines()
	for line in lines:
		words = line.split()
		out_words = []
		for word in words:
			if word not in vocab2id:
				out_words.append('<UNK>')
			else:
				out_words.append(word)
		if(use == 'target'):
			out_words = ['<GO>'] + out_words + ['<EOS>']
		out_f.write(' '.join(out_words) + '\n')
	for f in in_f, out_f:
		f.close()

def _seq2id(vocab2id, line):
	words = line.split()
	return [vocab2id.get(word, vocab2id['<UNK>']) for word in words]

def build_id_encode_data(in_name, lang):
	in_f = open(os.path.join(config.PROC_DIR, in_name), 'r')
	out_name = in_name + '.ids'
	out_f = open(os.path.join(config.PROC_DIR, out_name), 'w')
	vocab2id = _get_vocab2id('vocab.{}'.format(lang))
	lines = in_f.read().splitlines()
	for line in lines:
		ids = _seq2id(vocab2id, line)
		s_ids = [str(i) for i in ids]
		out_f.write(' '.join(s_ids) + '\n')
	for f in in_f, out_f:
		f.close()

def build_id_decode_data(in_name, mode, lang):
	in_f = open(os.path.join(config.PROC_DIR, in_name), 'r')
	out_name = '{}.{}.ids'.format(in_name, mode)
	out_f = open(os.path.join(config.PROC_DIR, out_name), 'w')
	vocab2id = _get_vocab2id('vocab.{}'.format(lang))
	lines = in_f.read().splitlines()
	for line in lines:
		ids = _seq2id(vocab2id, line)
		if mode == 'input':
			s_ids = [str(i) for i in ids[:-1]]
		else:
			s_ids = [str(i) for i in ids[1:]]
		out_f.write(' '.join(s_ids) + '\n')
	for f in in_f, out_f:
		f.close()

def embed_words(in_name, lang, sg = 1, window = 5, min_count = 5, negative = 5, n_iter = 1, save = True):
	in_f = open(os.path.join(config.PROC_DIR, in_name), 'r')
	lines = in_f.read().splitlines()
	sentences = [line.split() for line in lines]
	model = Word2Vec(sentences, sg = sg, size = config.EMBEDDING_DIM, window = window, min_count = min_count, negative = negative, iter = n_iter, compute_loss = True)
	print(lang.upper(), 'Loss', model.get_latest_training_loss())
	print('{} vocab size:'.format(lang.upper()), len(model.wv.vocab))
	if save == True:
		model.save(os.path.join(config.PROC_DIR, lang + '.bin'))
	in_f.close()

# def _check_words(words, phase):
# 	assert words[1] == '<UNK>' 
# 	if phase == 'decode':
# 		assert words[2] == '<GO>'
# 		assert words[3] == '<EOS>'

# def get_embedder(lang):
# 	embedder = [[0.0] * config.EMBEDDING_DIM]
# 	model = Word2Vec.load(os.path.join(config.PROC_DIR, lang + '.bin'))
# 	vocab2id = _get_vocab2id('vocab.{}'.format(lang))
# 	words = list(vocab2id.keys())
# 	# _check_words(words, phase)
# 	embedder = embedder + [model[word].tolist() for word in words[1:]]
# 	return embedder



def _sentence2id(sentence, vocab2id, tokenizer, rma):
	# sentence = sentence.lower()
	tokens = tokenizer(sentence, rma)
	ids = [vocab2id.get(token, vocab2id['<UNK>']) for token in tokens]
	return ids

def _get_id2vocab(in_name):
	with open(os.path.join(config.PROC_DIR, in_name), 'r') as in_f:
		vocab = in_f.read().splitlines()
	return dict(enumerate(vocab))

# if __name__ == "__main__":
	# parser = ArgumentParser()
	# parser.add_argument('--tokenize', '-t', action='store_true')
	# parser.add_argument('--check', '-c', action='store_true')
	# parser.add_argument('--build', '-b', action='store_true')
	# parser.add_argument('--embed', '-e', action='store_true')

	# args  = parser.parse_args()
	# if args.tokenize:
	# 	make_dir(config.PROC_DIR)
	# 	print('Tokenizing data ...')
	# 	tokenize_data('train.ja', 'ja')
	# 	tokenize_data('train.en', 'en')
	# 	tokenize_data('val.ja', 'ja')
	# 	tokenize_data('val.en', 'en')
	# 	# tokenize_data('test.ja', 'ja')
	# 	# tokenize_data('test.en', 'en')

	# if args.check:
	# 	print('Checking vocabulary...')
	# 	embed_words(in_name = 'train.ja', lang='ja', min_count = 1, save = True)
	# 	embed_words(in_name = 'train.en', lang='en', min_count = 1, save = True)

	# if args.build:
		# print('Building vocabulary ...')
		# build_vocab('ja')
		# build_vocab('en')

		# print('Adding special symbols...')
		# for f_name in 'train.ja', 'val.ja':
		# 	add_special_symbols(f_name, 'src', 'ja')
		# for f_name in 'train.en', 'val.en':
		# 	add_special_symbols(f_name, 'target', 'en')

		# for f_name in 'train.en', 'val.en':
		# 	add_special_symbols(f_name, 'src', 'en')
		# for f_name in 'train.ja', 'val.ja':
		# 	add_special_symbols(f_name, 'target', 'ja')

		# print('Building id data...')
		# for f_name in 'train.ja.src', 'val.ja.src':
		# 	build_id_encode_data(f_name, 'ja')
		# for f_name in 'train.ja.target', 'val.ja.target':
		# 	build_id_decode_data(f_name, 'input', 'ja')
		# 	build_id_decode_data(f_name, 'label', 'ja')

		# for f_name in 'train.en.src', 'val.en.src':
		# 	build_id_encode_data(f_name, 'en')
		# for f_name in 'train.en.target', 'val.en.target':
		# 	build_id_decode_data(f_name, 'input', 'en')
		# 	build_id_decode_data(f_name, 'label', 'en')

	# if args.embed:
	# 	print('Embedding words...')
	# 	embed_words(in_name = 'train.ja', lang = 'ja', sg = 1, window = 3, min_count = 10, negative = 20, n_iter = 5, save = True)
	# 	embed_words(in_name = 'train.en', lang = 'en', sg = 1, window = 3, min_count = 10, negative = 20, n_iter = 5, save = True)

