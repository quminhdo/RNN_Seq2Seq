import tensorflow as tf
import os
import numpy as np
import pdb
import dataUtils
import config
import modelHelper as helper

class TrainingModel:
	def __init__(self, encode_vocab_size, decode_vocab_size, data_batch):
		phase = 'train'
		enc_inputs = data_batch[0][0]
		enc_input_lens = data_batch[0][1]
		dec_inputs = data_batch[1][0]
		dec_input_lens = data_batch[1][1]
		dec_labels = data_batch[2][0]
		# dec_label_lens = data_batch[2][1]
		enc_embedder, dec_embedder = helper.build_embedders(encode_vocab_size, decode_vocab_size)
		# OTHERS
		self.global_step = tf.Variable(0, trainable = False)
		self.learning_rate = tf.Variable(config.LEARNING_RATE_0, dtype = tf.float32, trainable = False, name = 'learning_rate')
		self.lr1_op = self.learning_rate.assign(config.LEARNING_RATE_1)
		self.lr2_op = self.learning_rate.assign(config.LEARNING_RATE_2)
		self.lr3_op = self.learning_rate.assign(config.LEARNING_RATE_3)
		self.lr4_op = self.learning_rate.assign(config.LEARNING_RATE_4)
		self.lr5_op = self.learning_rate.assign(config.LEARNING_RATE_5)
		self.lr6_op = self.learning_rate.assign(config.LEARNING_RATE_6)
		self.decayed = tf.Variable(False, dtype = tf.bool, trainable = False, name = 'decayed')
		self.decay_op = self.decayed.assign(True)
		# EMBEDDING
		enc_inputs = tf.nn.embedding_lookup(enc_embedder, enc_inputs)
		dec_inputs = tf.nn.embedding_lookup(dec_embedder, dec_inputs)
		# ENCODER
		enc_outputs, enc_final_state = helper.build_encoder(phase, enc_inputs, enc_input_lens)
		# DECODER
		dec_outputs, dec_output_lens = helper.build_training_decoder(phase, decode_vocab_size, enc_outputs, enc_input_lens, enc_final_state, dec_embedder, dec_inputs, dec_input_lens)
		# LOSS COMPUTATION
		self.loss = helper.compute_loss(dec_output_lens, dec_labels, dec_outputs.rnn_output)

		# OPTIMIZER
		variables = tf.trainable_variables()
		print(len(variables))
		print(variables)
		gradients = tf.gradients(self.loss, variables)
		clipped_gradients, _ = tf.clip_by_global_norm(gradients, config.MAX_GRAD_NORM)
		optimizer = tf.train.AdamOptimizer(self.learning_rate)
		self.train_op = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step = self.global_step)


	def _update_learning_rate_by_loss(self, sess, loss):
		if loss < 2 and loss > 1:
			sess.run(self.lr1_op)
		if loss < 0.5 and loss > 0.1:
			sess.run(self.lr2_op)
		if loss < 0.02 and loss > 0.01:
			sess.run(self.lr3_op)
		if loss < 0.01 and loss > 0.005:
			sess.run(self.lr4_op)

	def _update_learning_rate_by_epoch(self, sess, epoch):
		if epoch > 22:
			sess.run(self.lr6_op)
		if epoch > 19 and epoch <= 22:
			sess.run(self.lr5_op)
		if epoch > 16 and epoch <= 19:
			sess.run(self.lr4_op)
		if epoch > 13 and epoch <= 16:
			sess.run(self.lr3_op)
		if epoch > 10 and epoch <= 13:
			sess.run(self.lr2_op)
		if epoch > 7 and epoch <= 10:
			sess.run(self.lr1_op)
		pass

	def _decay_learning_rate(self, sess, loss):
		if loss < 3 and self.decayed.eval() == False:
			self.learning_rate = tf.train.exponential_decay(learning_rate = config.LEARNING_RATE_0, global_step = self.global_step, decay_steps = config.DECAY_STEPS, decay_rate = config.DECAY_RATE)
			sess.run(self.decay_op)

	def train(self, sess):
		learning_rate = sess.run(self.learning_rate)
		loss, _, global_step= sess.run((self.loss, self.train_op, self.global_step))
		return loss, learning_rate, global_step

class ValidationModel:
	def __init__(self, encode_vocab_size, decode_vocab_size, data_batch):
		phase = 'validate'
		enc_inputs = data_batch[0][0]
		enc_input_lens = data_batch[0][1]
		dec_inputs = data_batch[1][0]
		dec_input_lens = data_batch[1][1]
		dec_labels = data_batch[2][0]
		# dec_label_lens = data_batch[2][1]
		enc_embedder, dec_embedder = helper.build_embedders(encode_vocab_size, decode_vocab_size)
		# EMBEDDING
		enc_inputs = tf.nn.embedding_lookup(enc_embedder, enc_inputs)
		dec_inputs = tf.nn.embedding_lookup(dec_embedder, dec_inputs)
		# ENCODER
		enc_outputs, enc_final_state = helper.build_encoder(phase, enc_inputs, enc_input_lens)
		# DECODER
		# if decoder_type == 'beamsearch':
		# 	dec_outputs, dec_output_lens = helper.build_beamsearch_decoder(phase, enc_outputs, enc_input_lens, enc_final_state, dec_embedder)
		dec_outputs, dec_output_lens = helper.build_training_decoder(phase, decode_vocab_size, enc_outputs, enc_input_lens, enc_final_state, dec_embedder, dec_inputs, dec_input_lens)
		# LOSS COMPUTATION
		self.loss = helper.compute_loss(dec_output_lens, dec_labels, dec_outputs.rnn_output)

	def validate(self, sess):
		loss = sess.run((self.loss))
		return loss

class InferenceModel:
	def __init__ (self, decoder_type, encode_vocab_size, decode_vocab_size, enc_input, enc_input_len):
		phase = 'infer'
		self.enc_input = enc_input
		self.enc_input_len = enc_input_len
		enc_embedder, dec_embedder = helper.build_embedders(encode_vocab_size, decode_vocab_size)
		# EMBEDDING_LAYER
		enc_input = tf.nn.embedding_lookup(enc_embedder, self.enc_input)
		# ENCODER
		enc_output, enc_final_state = helper.build_encoder(phase, enc_input, enc_input_len)
		# DECODER
		if decoder_type == 'beamsearch':
			dec_output, dec_output_len = helper.build_beamsearch_decoder(phase, decode_vocab_size, enc_output, enc_input_len, enc_final_state, dec_embedder)
			dec_output_ids = tf.squeeze(dec_output.predicted_ids)
			dec_output_ids = tf.transpose(dec_output_ids)
			self.output_id = dec_output_ids[0]
		else:
			dec_output, dec_output_len = helper.build_greedy_decoder(phase, decode_vocab_size, enc_output, enc_input_len, enc_final_state, dec_embedder)
			dec_output_id = dec_output.sample_id
			self.output_id = tf.squeeze(dec_output_id)

	def infer(self, sess, input_ids, input_len):
		output_id = sess.run(self.output_id, feed_dict = {self.enc_input:input_ids, self.enc_input_len:input_len})
		return output_id
