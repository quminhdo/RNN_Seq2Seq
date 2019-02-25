import tensorflow as tf

class Embedding_Layer:
    def __init__(self, vocab_size, embedding_dim, scope_name):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        with tf.variable_scope(scope_name) as scope:
            self.word_embedding=tf.get_variable("word_embedding", [vocab_size, embedding_dim])

    def __call__(self, id_seqs):
        return tf.gather(self.word_embedding, id_seqs)

    def linear(self, inputs):
        # inputs shape: [batch_size, max_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        max_len = tf.shape(inputs)[1]
        inputs = tf.reshape(inputs, [-1, self.embedding_dim])
        outputs = tf.matmul(inputs, self.word_embedding, transpose_b=True)
        return tf.reshape(outputs, [batch_size, max_len, self.vocab_size])

class Encoder:
    def __init__(self, cell_type, rnn_dim, rnn_layer_num, bidirectional):
        self.bidirectional = bidirectional
        if rnn_layer_num < 1:
            raise ValueError("rnn_layer_num must be greater than 0, but {} found instead.".format(rnn_layer_num))
        cell_dict = {"lstm":tf.nn.rnn_cell.LSTMCell, "gru":tf.nn.rnn_cell.GRUCell}
        if bidirectional:
            # build forward cell
            if rnn_layer_num > 1:
                self.cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_dict[cell_type](rnn_dim, name="fw_{}_{}".format(cell_type, i)) for i in range(rnn_layer_num)])
            else:
                self.cell_fw = cell_dict[cell_type](rnn_dim)
            # build backward cell
            if rnn_layer_num > 1:
                self.cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_dict[cell_type](rnn_dim, name="bw_{}_{}".format(cell_type, i)) for i in range(rnn_layer_num)])
            else:
                self.cell_bw = cell_dict[cell_type](rnn_dim)
        else:
            if rnn_layer_num > 1:
                self.cell = tf.nn.rnn_cell.MultiRNNCell([cell_dict[cell_type](rnn_dim, name="{}_{}".format(cell_type, i)) for i in range(rnn_layer_num)])
            else:
                self.cell = cell_dict[cell_type](rnn_dim)

    def __call__(self, inputs, sequence_length):
        with tf.variable_scope("encoder") as scope:
            if self.bidirectional:
                outputs, final_states = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, inputs, sequence_length=sequence_length, dtype=tf.float32)
                outputs = tf.concat(outputs, axis=-1)
            else:
                outputs, final_states = tf.nn.dynamic_rnn(self.cell, inputs, sequence_length=sequence_length, dtype=tf.float32)
        return outputs, final_states

class Decoder:
    def __init__(self, cell_type, rnn_dim, rnn_layer_num):
        if rnn_layer_num < 1:
            raise ValueError("rnn_layer_num must be greater than 0, but {} found instead.".format(rnn_layer_num))
        cell_dict = {"lstm":tf.nn.rnn_cell.LSTMCell, "gru":tf.nn.rnn_cell.GRUCell}
        if rnn_layer_num > 1:
            self.cell = tf.nn.rnn_cell.MultiRNNCell([cell_dict[cell_type](rnn_dim, name="{}_{}".format(cell_type, i)) for i in range(rnn_layer_num)])
        else:
            self.cell = cell_dict[cell_type](rnn_dim)

    def __call__(self, inputs, sequence_length, initial_state=None):
        with tf.variable_scope("decoder") as scope:
            outputs, final_states = tf.nn.dynamic_rnn(self.cell, inputs, sequence_length=sequence_length, initial_state=initial_state, dtype=tf.float32)
        return outputs, final_states
    
class Attention_Layer:
    def __init__(self, attention_dim):
        self.query_proj_layer = tf.layers.Dense(attention_dim, use_bias=False)
        self.key_proj_layer = tf.layers.Dense(attention_dim, use_bias=False)
    
    def __call__(self, encoder_outputs, decoder_outputs, attention_mask):
        q = self.query_proj_layer(decoder_outputs) # shape [batch_size, decoder_max_len, attention_dim]
        k = self.key_proj_layer(encoder_outputs) # shape [batch_size, encoder_max_len, attention_dim]
        v = encoder_outputs
        qk = tf.matmul(q, k, transpose_b=True) 
        attention_weights = tf.nn.softmax(qk + attention_mask) # shape [batch_size, decoder_max_len, encoder_max_len]
        context = tf.matmul(attention_weights, v) # shape [batch_size, decoder_max_len, rnn_dim]
        outputs = tf.concat((context, decoder_outputs), -1) # shape [batch_size, decoder_max_len, 2*rnn_dim]
        return outputs, attention_weights
    
