import tensorflow as tf
import layers

class RNN_Seq2Seq:
    def __init__(self, 
        source_vocab_size, 
        target_vocab_size, 
        embedding_dim, 
        cell_type, 
        rnn_dim, 
        encoder_rnn_layer_num, 
        bidirectional, 
        decoder_rnn_layer_num, 
        attention_dim, 
        go_id, 
        eos_id, 
        pad_id):
        self.GO = go_id
        self.EOS = eos_id
        self.PAD = pad_id
        self.TARGET_VOCAB_SIZE = target_vocab_size
        self.rnn_dim = rnn_dim
        self.source_embedding_layer = layers.Embedding_Layer(source_vocab_size, embedding_dim, "source")
        self.target_embedding_layer = layers.Embedding_Layer(target_vocab_size, embedding_dim, "target")
        self.encoder = layers.Encoder(cell_type, rnn_dim, encoder_rnn_layer_num, bidirectional)
        self.decoder = layers.Decoder(cell_type, rnn_dim, decoder_rnn_layer_num)
        self.attention_layer = layers.Attention_Layer(attention_dim)
        self.projection_layer = tf.layers.Dense(embedding_dim, use_bias=False)

    def __call__(self, source_id_seqs, target_id_seqs=None, length_penalty_weight=None, coverage_penalty_weight=None, beam_size=None):
        attention_mask = self.get_attention_mask(source_id_seqs)
        encoder_outputs, encoder_states = self.encode(source_id_seqs)
        if target_id_seqs is None:
            if beam_size is None:
                fed_inputs, attention_weights, outputs  = self.greedy_decode(encoder_outputs, encoder_states, attention_mask)
            else:
                fed_inputs, attention_weights, outputs = self.beam_search_decode(encoder_outputs, encoder_states, attention_mask, beam_size, coverage_penalty_weight, length_penalty_weight)
            output_dict = {"fed_inputs": fed_inputs, "attention_weights": attention_weights, "outputs": outputs}
        else:
            logits, outputs = self.decode_with_teacher_forcing(encoder_outputs, encoder_states, target_id_seqs, attention_mask)
            output_dict = {"logits": logits, "outputs": outputs}
        return output_dict

    def encode(self, id_seqs):
        sequence_length = self.get_sequence_length(id_seqs)
        inputs = self.source_embedding_layer(id_seqs)
        outputs, states = self.encoder(inputs, sequence_length)
        return outputs, states
    
    def decode_with_teacher_forcing(self, encoder_outputs, encoder_states, id_seqs, attention_mask):
        sequence_length = self.get_sequence_length(id_seqs)
        inputs = self.target_embedding_layer(id_seqs)
        outputs, states = self.decoder(inputs, sequence_length)
        outputs, attention_weights = self.attention_layer(encoder_outputs, outputs, attention_mask) # shape [batch_size, decoder_max_len, 2*rnn_dim]
        outputs = self.projection_layer(outputs) # shape [batch_size, decoder_max_len, embedding_dim]
        logits = self.target_embedding_layer.linear(outputs) # shape [batch_size, decoder_max_len, target_vocab_size]
        outputs = tf.argmax(tf.nn.softmax(logits), axis=-1, output_type=tf.int32)
        return logits, outputs

    def greedy_decode(self, encoder_outputs, encoder_states, attention_mask, extra_decode_length=20):
        batch_size = tf.shape(encoder_outputs)[0]
        encode_length = tf.shape(encoder_outputs)[1]
        MAX_LENGTH = encode_length + extra_decode_length
        initial_variables = {
            "ID_SEQS" : tf.fill([batch_size, 1], self.GO),
            "ATTENTION_WEIGHTS": tf.zeros([batch_size, 1, encode_length]),
            "OUTPUTS" : tf.zeros([batch_size, 1], tf.int32),
            "FINISHED" : tf.zeros([batch_size], tf.int32),
            "INDEX" : tf.constant(0)
        }
        batch_size_dim = encoder_outputs.shape[0]
        encode_length_dim = encoder_outputs.shape[1]
        variables_shape = {
            "ID_SEQS" : tf.TensorShape([batch_size_dim, None]),
            "ATTENTION_WEIGHTS" : tf.TensorShape([batch_size_dim, None, encode_length_dim]),
            "OUTPUTS": tf.TensorShape([batch_size_dim, None]),
            "FINISHED": tf.TensorShape([batch_size_dim]),
            "INDEX": tf.TensorShape([]) 
        }
        def continue_decode(fed_inputs, attention_weights, outputs, finished, i):
            return tf.logical_and(tf.less(tf.reduce_sum(finished), tf.size(finished)), tf.less(tf.shape(outputs)[1], MAX_LENGTH))

        def step(fed_inputs, attention_weights, outputs, finished, i):
            i += 1
            cur_outputs = outputs[:, -1] # shape [batch_size]
            next_ids = (1 - finished) * cur_outputs + finished * self.EOS
            fed_inputs = tf.cond(tf.equal(i, 1), lambda: fed_inputs, lambda: tf.concat([fed_inputs, tf.expand_dims(next_ids, -1)], -1))
            sequence_length = self.get_sequence_length(fed_inputs)
            inputs = self.target_embedding_layer(fed_inputs)
            decoder_outputs, states = self.decoder(inputs, sequence_length)
            attention_layer_outputs, attention_weights = self.attention_layer(encoder_outputs, decoder_outputs, attention_mask) # shape [batch_size, decoder_cur_len, 2*rnn_dim]
            projected_outputs = self.projection_layer(attention_layer_outputs) # shape [batch_size, decoder_cur_len, embedding_dim]
            logits = self.target_embedding_layer.linear(projected_outputs) # shape [batch_size, decoder_cur_len, target_vocab_size]
            outputs = tf.argmax(tf.nn.softmax(logits), axis=-1, output_type=tf.int32) # shape [batch_size, decoder_cur_len]
            cur_outputs = outputs[:, -1] # shape [batch_size]
            finished = tf.maximum(finished, tf.cast(tf.equal(cur_outputs, self.EOS), finished.dtype))
            return fed_inputs, attention_weights, outputs, finished, i

        fed_inputs, attention_weights, outputs, _, _ = tf.while_loop(continue_decode, step, list(initial_variables.values()), shape_invariants=list(variables_shape.values()))
        return fed_inputs, attention_weights, outputs
        
    def beam_search_decode(self, encoder_outputs, encoder_states, attention_mask, beam_size, coverage_penalty_weight, length_penalty_weight, extra_decode_length=20):
        beam_size = beam_size
        coverage_penalty_weight = coverage_penalty_weight
        length_penalty_weight = length_penalty_weight
        batch_size = tf.shape(encoder_outputs)[0]
        encode_length = tf.shape(encoder_outputs)[1]
#        rnn_dim = tf.shape(encoder_outputs)[2]
        rnn_dim = self.rnn_dim
        MAX_LENGTH = encode_length + extra_decode_length
        initial_variables = {
            "FED_INPUTS" : tf.fill([batch_size*beam_size, 1], self.GO),
            "ATTENTION_WEIGHTS": tf.zeros([batch_size*beam_size, 1, encode_length]),
            "OUTPUTS" : tf.zeros([batch_size*beam_size, 1], tf.int32),
            "FINISHED" : tf.zeros([batch_size*beam_size], tf.int32),
            "SCORES": tf.zeros([batch_size, beam_size*self.TARGET_VOCAB_SIZE]),
            "INDEX" : tf.constant(0)
        }
        batch_size_dim = encoder_outputs.shape[0]
        encode_length_dim = encoder_outputs.shape[1]
        variables_shape = {
            "FED_INPUTS" : tf.TensorShape([batch_size_dim*beam_size, None]),
            "ATTENTION_WEIGHTS" : tf.TensorShape([batch_size_dim*beam_size, None, encode_length_dim]),
            "OUTPUTS": tf.TensorShape([batch_size_dim*beam_size, None]),
            "FINISHED": tf.TensorShape([batch_size_dim*beam_size]),
            "SCORES": tf.TensorShape([batch_size_dim, beam_size*self.TARGET_VOCAB_SIZE]),
            "INDEX": tf.TensorShape([]) 
        }
        encoder_outputs = tf.tile(encoder_outputs, [beam_size, 1, 1])
        encoder_outputs = tf.reshape(encoder_outputs, [beam_size, batch_size, encode_length, rnn_dim])
        encoder_outputs = tf.transpose(encoder_outputs, [1, 0, 2, 3])
        encoder_outputs = tf.reshape(encoder_outputs, [batch_size*beam_size, encode_length, rnn_dim])

        def gather_nd(params, indices):
            indices_shape = tf.shape(indices)
            indices_size = tf.size(indices)
            i = tf.stack([tf.range(indices_size), tf.reshape(indices, [indices_size])], axis=1)
            p = tf.reshape(params, [indices_size, -1])
            return tf.reshape(tf.gather_nd(p, i), indices_shape)

        def compute_score(i, prev_outputs, logits, sequence_length, attention_weights, fed_inputs):
            cur_softmax = tf.nn.softmax(logits[:, -1:, :])
            cur_probabilities = tf.transpose(cur_softmax, [0, 2, 1])
            def generate_probabilities():
                decoded_length = tf.shape(prev_outputs)[1]
                prev_softmax = tf.nn.softmax(logits[:, :-1, :])
                prev_probabilities = gather_nd(prev_softmax, prev_outputs)
                prev_probabilities = tf.tile(prev_probabilities, [1, self.TARGET_VOCAB_SIZE])
                prev_probabilities = tf.reshape(prev_probabilities, [batch_size*beam_size, self.TARGET_VOCAB_SIZE, decoded_length])
                return tf.concat([prev_probabilities, cur_probabilities], -1)
            probabilities = tf.cond(tf.equal(i, 1), lambda: cur_probabilities, lambda: generate_probabilities())
            probabilities = tf.reduce_prod(probabilities, axis=-1) # shape [batch_size*beam_size, TARGET_VOCAB_SIZE]

            length_penalty = tf.pow((5.0 + tf.to_float(sequence_length))/(5.0 + 1.0), length_penalty_weight)
            
            unpadded_pos = tf.cast(tf.not_equal(fed_inputs, self.PAD), tf.float32)
            masked_attention_weights = tf.expand_dims(unpadded_pos, -1)*attention_weights
            coverage_penalty = coverage_penalty_weight*tf.reduce_sum(tf.log(tf.minimum(tf.reduce_sum(masked_attention_weights, 1), 1.0)), -1) # shape [batch_size*beam_size]

            scores = tf.log(probabilities)/tf.expand_dims(length_penalty, -1) + tf.expand_dims(coverage_penalty, -1)
            scores = tf.reshape(scores, [batch_size, beam_size*self.TARGET_VOCAB_SIZE])
            return scores

        def generate_outputs(i, outputs, scores, _beam_size):
            top_scores, indices = tf.nn.top_k(scores, k=_beam_size) # shape [batch_size, beam_size]
            top_beams = tf.floordiv(indices, self.TARGET_VOCAB_SIZE)
            top_beams = _beam_size*tf.expand_dims(tf.range(batch_size), -1)+top_beams
            top_beams = tf.reshape(top_beams, [batch_size*_beam_size])
            top_ids = tf.floormod(indices, self.TARGET_VOCAB_SIZE)
            top_ids = tf.reshape(top_ids, [batch_size*_beam_size, 1])
            outputs = tf.cond(tf.equal(i, 1), lambda: top_ids, lambda: tf.concat([tf.gather(outputs, top_beams), top_ids], -1))
            return outputs

        def continue_decode(fed_inputs, attention_weights, outputs, finished, scores, i):
            return tf.logical_and(tf.less(tf.reduce_sum(finished), tf.size(finished)), tf.less(tf.shape(outputs)[1], MAX_LENGTH))
            
        def step(fed_inputs, attention_weights, outputs, finished, scores, i):
            i += 1
            cur_outputs = outputs[:, -1] # shape [batch_size*beam_size]
            next_ids = (1 - finished) * cur_outputs + finished * self.PAD
            fed_inputs = tf.cond(tf.equal(i, 1), lambda: fed_inputs, lambda: tf.concat([fed_inputs, tf.expand_dims(next_ids, -1)], -1))
            sequence_length = self.get_sequence_length(fed_inputs)
            inputs = self.target_embedding_layer(fed_inputs)
            decoder_outputs, states = self.decoder(inputs, sequence_length)
            attention_layer_outputs, attention_weights = self.attention_layer(encoder_outputs, decoder_outputs, attention_mask) # shape [batch_size*beam_size, decoder_cur_len, 2*rnn_dim]
            projected_outputs = self.projection_layer(attention_layer_outputs) 
            logits = self.target_embedding_layer.linear(projected_outputs) # shape [batch_size*beam_size, decoder_cur_len, target_vocab_size]
            scores = compute_score(i, outputs, logits, sequence_length, attention_weights, fed_inputs) 
            outputs = generate_outputs(i, outputs, scores, beam_size)
            cur_outputs = outputs[:, -1]
            finished = tf.maximum(finished, tf.cast(tf.equal(cur_outputs, self.EOS), finished.dtype))
            return fed_inputs, attention_weights, outputs, finished, scores, i
        fed_inputs, attention_weights, outputs, _, scores, i = tf.while_loop(continue_decode, step, list(initial_variables.values()), shape_invariants=list(variables_shape.values()))
        outputs = generate_outputs(i, outputs[:, :-1], scores, 1)
        return fed_inputs, attention_weights, outputs

    def get_sequence_length(self, id_seqs): # shape [batch_size, max_len]
        unpadded_pos = tf.cast(tf.not_equal(id_seqs, self.PAD), tf.int32)
        return tf.reduce_sum(unpadded_pos, -1) # shape [batch_size]

    def get_attention_mask(self, id_seqs, neg_inf=-1e15): # shape: [batch_size, max_len]
        padded_pos = tf.cast(tf.equal(id_seqs, self.PAD), tf.float32)
        return tf.expand_dims(padded_pos * neg_inf, axis=1) # shape [batch_size, 1, max_len]
