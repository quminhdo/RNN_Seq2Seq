import tensorflow as tf
import os
import numpy as np
import pdb
import dataUtils
import config
import rnn
from layernormLSTM import LayerNormBasicLSTMCell
from gnmt import GNMTAttentionMultiCell
from gnmt import TestAttentionMultiCell
# from decoder import BasicDecoder

def build_embedders(encode_vocab_size, decode_vocab_size, ):
	if config.USE_TRAINED_EMBEDDING:
		enc_embedder = tf.convert_to_tensor(dataUtils.get_embedder('encode'))
		dec_embedder = tf.convert_to_tensor(dataUtils.get_embedder('decode'))
	else:
		# initializer = tf.random_uniform_initializer
		initializer = tf.random_normal_initializer(mean=0.0, stddev=config.STDDEV)
		enc_embedder = tf.get_variable(name='encoder_embedder', shape=[encode_vocab_size, config.EMBEDDING_DIM], dtype=tf.float32, initializer=initializer)
		dec_embedder = tf.get_variable(name='decoder_embedder', shape=[decode_vocab_size, config.EMBEDDING_DIM], dtype=tf.float32, initializer=initializer)
	return enc_embedder, dec_embedder

def _build_single_cell(cell_type, dropout):
	if cell_type == 'lstm':
		cell = tf.contrib.rnn.LSTMCell(num_units=config.HIDDEN_SIZE)
	if cell_type == 'norm_lstm':
		cell = LayerNormBasicLSTMCell(num_units=config.HIDDEN_SIZE)
	if cell_type == 'gru':
		cell = tf.contrib.rnn.GRUCell(config.HIDDEN_SIZE)
	if cell_type == 'vanilla':
		cell = tf.contrib.rnn.BasicRNNCell(config.HIDDEN_SIZE)
	if dropout:
		print('Dropout...')
		cell = tf.contrib.rnn.DropoutWrapper(cell = cell, input_keep_prob = 1 - config.DROPOUT)
	return cell

def _build_bidirectional_layers(cell_type, inputs, input_lens, dropout):
	cell_fw = _build_single_cell(cell_type, dropout)
	cell_bw = _build_single_cell(cell_type, dropout)
	outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw, cell_bw = cell_bw, inputs = inputs, sequence_length = input_lens, dtype = tf.float32, time_major = config.TIME_MAJOR)
	#outputs, output_states = rnn.bidirectional_dynamic_rnn(cell_fw = cell_fw, cell_bw = cell_bw, inputs = inputs, sequence_length = input_lens, dtype = tf.float32, time_major = config.TIME_MAJOR, concatenate=config.CONCATENATE)
	return tf.concat(outputs, -1), output_states

def _build_multi_cells(cell_type, n_layers, dropout):
	if n_layers == 1:
		return _build_single_cell(cell_type, dropout)
	cells = []
	for _ in range(n_layers):
		cells.append(_build_single_cell(cell_type, dropout))
	return tf.contrib.rnn.MultiRNNCell(cells)

def build_encoder(phase, inputs, input_lens):
	if config.TIME_MAJOR:
		inputs = tf.transpose(inputs, [1, 0, 2])
	n_bi_layers = config.ENCODER_BI_LAYERS
	n_uni_layers = config.ENCODER_LAYERS - 2*n_bi_layers
	assert n_uni_layers >= 0
	dropout = True if phase == 'train' and config.DROPOUT > 0 else False
	final_states = []
	outputs = None
	for _ in range(n_bi_layers):
		outputs, _final_states = _build_bidirectional_layers(config.ENCODER_CELL_TYPE, inputs, input_lens, dropout)
		final_states += _final_states
	if n_uni_layers > 0:
		cell = _build_multi_cells(config.ENCODER_CELL_TYPE, n_uni_layers, dropout)
		_inputs = inputs if outputs == None else outputs
		outputs, _final_states = tf.nn.dynamic_rnn(cell = cell, inputs = _inputs, sequence_length = input_lens, dtype = tf.float32, time_major = config.TIME_MAJOR)
		if n_uni_layers == 1:
			final_states.append(_final_states)
		else:
			final_states += _final_states
	if config.TIME_MAJOR:
		outputs = tf.transpose(outputs, [1, 0, 2])
	if(len(final_states) > 1):
		return outputs, tuple(final_states)
	return outputs, final_states[0]

def _build_attention_cell(cell, enc_outputs, enc_output_lens):
	if config.ATTENTION == 'luong':
		attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units = config.HIDDEN_SIZE, memory = enc_outputs, memory_sequence_length = enc_output_lens, dtype = tf.float32)
	if config.ATTENTION == 'bahdanau':
		attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(normalize=config.NORMALIZE, num_units = config.HIDDEN_SIZE, memory = enc_outputs, memory_sequence_length = enc_output_lens, dtype = tf.float32)
	attention_cell = tf.contrib.seq2seq.AttentionWrapper(cell = cell, attention_mechanism = attention_mechanism, attention_layer_size = config.HIDDEN_SIZE)
	# initial_state = attention_cell.zero_state(tf.size(enc_output_lens), tf.float32)
	# if config.PASS_STATE:
	# 	initial_state = initial_state.clone(cell_state = passed_state)
	return attention_cell

def _build_decoder_cell(phase, enc_outputs, enc_output_lens, enc_final_state):
	dropout = True if phase == 'train' and config.DROPOUT > 0 else False
	batch_size = tf.size(enc_output_lens)
	if config.DECODER == 'baseline':		
		cell = _build_multi_cells(config.DECODER_CELL_TYPE, config.DECODER_LAYERS, dropout)
		#initial_state = enc_final_state
		# print('enc_final_state', enc_final_state)
		#enc_final_state = enc_final_state[0]
		if config.ATTENTION:
			cell = _build_attention_cell(cell, enc_outputs, enc_output_lens)
		initial_state = cell.zero_state(batch_size, tf.float32)
		if config.PASS_STATE:
			if config.ATTENTION:
				initial_state.clone(cell_state=enc_final_state)
			else:
				initial_state = enc_final_state

	elif config.DECODER == 'gnmt':
		cell_list = []
		initial_state_list = []
		for _ in range(config.DECODER_LAYERS):
			cell_list.append(_build_single_cell(config.DECODER_CELL_TYPE, dropout))
		attention_cell = cell_list.pop(0)
		attention_cell = _build_attention_cell(attention_cell, enc_outputs, enc_output_lens)
		cell = GNMTAttentionMultiCell(attention_cell, cell_list)
		# batch_size = tf.size(enc_output_lens)
		if config.PASS_STATE:
			initial_state = tuple(zs.clone(cell_state=es) 
				if isinstance(zs, tf.contrib.seq2seq.AttentionWrapperState) else es for zs, es in zip(cell.zero_state(batch_size, tf.float32), enc_final_state))
		else:
			initial_state = cell.zero_state(batch_size, tf.float32)
	elif config.DECODER == 'test':
		if config.ENCODER_BI_LAYERS > 0 and config.PASS_STATE:
			assert config.ENCODER_LAYERS > 2* config.ENCODER_BI_LAYERS
		cell_list = []
		initial_state_list = []
		for _ in range(config.DECODER_LAYERS):
			cell_list.append(_build_single_cell(config.DECODER_CELL_TYPE, dropout))
		attention_cell = cell_list.pop(0)
		attention_cell = _build_attention_cell(attention_cell, enc_outputs, enc_output_lens)
		cell = TestAttentionMultiCell(attention_cell, cell_list, config.APPLY_ATTENTION_TO_ALL)
		# batch_size = tf.size(enc_output_lens)
		initial_state = list(cell.zero_state(batch_size, tf.float32))
		if config.PASS_STATE:
			initial_state[0] = initial_state[0].clone(cell_state=enc_final_state[-1]) if isinstance(initial_state[0], tf.contrib.seq2seq.AttentionWrapperState) else enc_final_state[-1]
		initial_state = tuple(initial_state)
	return cell, initial_state

def build_training_decoder(phase, decode_vocab_size, enc_outputs, enc_output_lens, enc_final_state, dec_embedder, dec_inputs, dec_input_lens):
	cell, initial_state = _build_decoder_cell(phase, enc_outputs, enc_output_lens, enc_final_state)
	helper = tf.contrib.seq2seq.TrainingHelper(inputs = dec_inputs, sequence_length = dec_input_lens)
	projection_layer = tf.layers.Dense(units = decode_vocab_size, use_bias = False)
	decoder = tf.contrib.seq2seq.BasicDecoder(cell = cell, helper = helper, initial_state = initial_state, output_layer = projection_layer)
	# decoder = BasicDecoder(cell = cell, helper = helper, initial_state = initial_state, target_embedding=dec_embedder)
	dec_outputs, _, dec_output_lens = tf.contrib.seq2seq.dynamic_decode(decoder = decoder, impute_finished = True, maximum_iterations = None)
	return dec_outputs, dec_output_lens

def build_greedy_decoder(phase, decode_vocab_size, enc_outputs, enc_output_lens, enc_final_state, dec_embedder):
	cell, initial_state = _build_decoder_cell(phase, enc_outputs, enc_output_lens, enc_final_state)
	start_tokens = [2] if phase == 'infer' else [2]*config.BATCH_SIZE
	helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embedder, start_tokens, 3)
	projection_layer = tf.layers.Dense(units = decode_vocab_size, use_bias = False)
	decoder = tf.contrib.seq2seq.BasicDecoder(cell = cell, helper = helper, initial_state = initial_state, output_layer = projection_layer)
	# decoder = BasicDecoder(cell = cell, helper = helper, initial_state = initial_state, target_embedding=dec_embedder)
	dec_outputs, _, dec_output_lens = tf.contrib.seq2seq.dynamic_decode(decoder = decoder, impute_finished = True, maximum_iterations = config.MAX_LENGTH)
	return dec_outputs, dec_output_lens

def build_beamsearch_decoder(phase, decode_vocab_size, enc_outputs, enc_output_lens, enc_final_state, dec_embedder):
	input_shape = enc_outputs.get_shape()
	input_size = 1 if phase == 'infer' else input_shape[0]
	tiled_enc_outputs = tf.contrib.seq2seq.tile_batch(enc_outputs, config.BEAM_WIDTH)
	tiled_enc_output_lens = tf.contrib.seq2seq.tile_batch(enc_output_lens, config.BEAM_WIDTH)
	tiled_enc_final_state = tf.contrib.seq2seq.tile_batch(enc_final_state, config.BEAM_WIDTH)
	cell, initial_state = _build_decoder_cell(phase, tiled_enc_outputs, tiled_enc_output_lens, tiled_enc_final_state)
	projection_layer = tf.layers.Dense(units = decode_vocab_size, use_bias = False)
	decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell = cell, embedding = dec_embedder, start_tokens = [2]*input_size, end_token = 3, initial_state = initial_state, beam_width = config.BEAM_WIDTH, output_layer = projection_layer, length_penalty_weight = config.LENGTH_PENALTY)
	dec_outputs, _, dec_output_lens = tf.contrib.seq2seq.dynamic_decode(decoder = decoder, impute_finished = False, maximum_iterations = config.MAX_LENGTH)
	return dec_outputs, dec_output_lens

def compute_loss(masking_lens, labels, logits):
	masking_weights = tf.sequence_mask(lengths = masking_lens, dtype = tf.float32)
	crossents = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = logits)
	loss = tf.reduce_sum(crossents*masking_weights)/config.BATCH_SIZE
	return loss

