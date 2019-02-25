import os
import time
import tensorflow as tf
import utils
import config
from rnn_seq2seq import RNN_Seq2Seq

class Training_Model:
    def __init__(self, data_path, languages, model_path, graph):
        self.model_params = config.MODEL_PARAMS
        self.train_params = config.TRAIN_PARAMS
        self.base_params = config.BASE_PARAMS
        self.data_path = data_path
        self.languages = languages
        self.model_path = model_path
        self.PAD = self.base_params["pad_id"]
        self.GO = self.base_params["go_id"]
        self.EOS = self.base_params["eos_id"]
        self.UNK = self.base_params["unk_id"]
        self.source_converter = self.get_converter(languages["source"])
        self.target_converter = self.get_converter(languages["target"])
        epoch = tf.get_variable(name="epoch", initializer=0, trainable=False)
        self.update_epoch = tf.assign_add(epoch, 1)
        self.learning_rate = self.train_params["learning_rate"] * tf.pow(tf.to_float(self.train_params["decay_rate"]), tf.to_float((epoch - 1) // self.train_params["decay_step"]))
        self.sess = tf.Session(graph=graph)
        # Prepare inputs and target outputs 
        dataset = self.prepare_dataset()
        self.iterator = self.get_iterator(dataset)
        data_batch = self.iterator.get_next()
        source_inputs, target_inputs, self.target_outputs = data_batch[0], data_batch[1], data_batch[2]
        self.target_inputs = target_inputs
        # Build model
        vocab_size = self.get_vocab_size()
        seq2seq = RNN_Seq2Seq(
        vocab_size["source"], 
        vocab_size["target"], 
        self.model_params["embedding_dim"], 
        self.model_params["cell_type"], 
        self.model_params["rnn_dim"],
        self.model_params["encoder_rnn_layer_num"],
        self.model_params["bidirectional"],
        self.model_params["decoder_rnn_layer_num"],
        self.model_params["attention_dim"],
        self.base_params["go_id"],
        self.base_params["eos_id"],
        self.base_params["pad_id"]
        )
        output_dict = seq2seq(source_inputs, target_inputs)
        logits = output_dict["logits"]
        self.outputs = output_dict["outputs"]
        crossents = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_outputs, logits=logits)
        batch_size = tf.to_float(tf.shape(logits)[0])
        mask_weight = self.get_mask_weight()
        self.loss = tf.reduce_sum(mask_weight*crossents)/batch_size
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)
        # save and restore
        self.saver = tf.train.Saver(max_to_keep=20)
        self.sess.run(tf.tables_initializer())
        self.check_restore_params()

    def __call__(self): 
        self.sess.run(self.iterator.initializer)
        total_loss = 0
        epoch = self.sess.run(self.update_epoch)
        n_batch = 0
        while True:
            try:
                target_inputs, outputs, target_outputs, learning_rate, batch_loss, _ = self.sess.run((self.target_inputs, self.outputs, self.target_outputs, self.learning_rate, self.loss, self.train_op))
                n_batch += 1
                total_loss += batch_loss
                #if n_batch % 10 == 1 and n_batch < 1000:
                #    print("Batch {} loss {}".format(n_batch, batch_loss))
            except tf.errors.OutOfRangeError:
                log = "Epoch {}:\nTime: {}\nTraining loss: {}\nLearning rate: {}\n".format(epoch, time.asctime(time.localtime(time.time())), total_loss/n_batch, learning_rate)
                print("No of batches:", n_batch)
#                self.check_training(outputs, target_outputs, target_inputs)
                self.saver.save(self.sess, os.path.join(self.model_path, "epoch_%d"%epoch))
                break;
        return log, epoch

    def get_converter(self, language):
        return utils.Word_Id_Converter(os.path.join(self.data_path, "vocab_{}".format(language)), self.UNK)

    def get_mask_weight(self):
        unpadded_pos = tf.cast(tf.not_equal(self.target_outputs, self.PAD), tf.float32) 
        return unpadded_pos

    def check_training(self, output, target, inputs=None):
        if inputs is not None:
            print("Input:")
            print(inputs)
        print("Output:")
        print(output)
        print("Target:")
        print(target)

    def get_vocab_size(self):
        vocab_size = {}
        for key, value in self.languages.items():
            vocab_size[key] = utils.get_file_size(os.path.join(self.data_path, "vocab_{}".format(value)))
        return vocab_size
    
    def check_restore_params(self):
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("Initializing parameters...")
            self.sess.run(tf.global_variables_initializer())
         
    def prepare_dataset(self):
        source_data_file = os.path.join(self.data_path, "train_{}".format(self.languages["source"]))
        target_data_file = os.path.join(self.data_path, "train_{}".format(self.languages["target"]))
        return utils.prepare_dataset(source_data_file, target_data_file, self.source_converter, self.target_converter, self.GO, self.EOS, self.UNK)

    def get_iterator(self, dataset):
        with tf.device("/cpu:0"):
            pad_value = tf.cast(self.PAD, tf.int32)
            dataset = dataset.shuffle(4000000, seed=0, reshuffle_each_iteration=True)
            dataset = dataset.padded_batch(self.train_params["batch_size"], ([None], [None], [None]), (pad_value, pad_value, pad_value), drop_remainder=False)
            return dataset.make_initializable_iterator()

    def update_epoch(self):
        return self.sess.run(tf.assign_add(self.epoch, 1))

class Validating_Model:
    def __init__(self, data_path, languages, model_path, graph):
        self.model_params = config.MODEL_PARAMS
        self.base_params = config.BASE_PARAMS
        self.data_path = data_path
        self.languages = languages
        self.model_path = model_path
        self.PAD = self.base_params["pad_id"]
        self.GO = self.base_params["go_id"]
        self.EOS = self.base_params["eos_id"]
        self.UNK = self.base_params["unk_id"]
        self.source_converter = self.get_converter(languages["source"])
        self.target_converter = self.get_converter(languages["target"])
        self.sess = tf.Session(graph=graph)
        # Prepare data 
        dataset = self.prepare_dataset()
        self.iterator = self.get_iterator(dataset)
        data_batch = self.iterator.get_next()
        source_inputs, target_inputs, self.target_outputs = data_batch[0], data_batch[1], data_batch[2]
        # Build model
        vocab_size = self.get_vocab_size()
        seq2seq = RNN_Seq2Seq(
        vocab_size["source"], 
        vocab_size["target"], 
        self.model_params["embedding_dim"], 
        self.model_params["cell_type"], 
        self.model_params["rnn_dim"],
        self.model_params["encoder_rnn_layer_num"],
        self.model_params["bidirectional"],
        self.model_params["decoder_rnn_layer_num"],
        self.model_params["attention_dim"],
        self.base_params["go_id"],
        self.base_params["eos_id"],
        self.base_params["pad_id"]
        )
        output_dict = seq2seq(source_inputs, target_inputs)
        logits = output_dict["logits"]
        crossents = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_outputs, logits=logits)
        batch_size = tf.to_float(tf.shape(logits)[0])
        mask_weight = self.get_mask_weight()
        self.loss = tf.reduce_sum(mask_weight*crossents)/batch_size
        # Save and restore
        self.saver = tf.train.Saver()
        self.sess.run(tf.tables_initializer())

    def __call__(self): 
        self.restore_params()
        self.sess.run(self.iterator.initializer)
        total_loss = 0
        n_batch = 0
        while True:
            try:
                batch_loss = self.sess.run((self.loss))
                total_loss += batch_loss
                n_batch += 1
            except tf.errors.OutOfRangeError:
                log = "Validation loss: {}\n".format(total_loss/n_batch)
                break;
        return log

    def get_converter(self, language):
        return utils.Word_Id_Converter(os.path.join(self.data_path, "vocab_{}".format(language)), self.UNK)

    def get_mask_weight(self):
        unpadded_pos = tf.cast(tf.not_equal(self.target_outputs, self.PAD), tf.float32) 
        return unpadded_pos

    def get_vocab_size(self):
        vocab_size = {}
        for key, value in self.languages.items():
            vocab_size[key] = utils.get_file_size(os.path.join(self.data_path, "vocab_{}".format(value)))
        return vocab_size

    def restore_params(self):
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("No saved parameters found")
         
    def prepare_dataset(self):
        source_data_file = os.path.join(self.data_path, "validate_{}".format(self.languages["source"]))
        target_data_file = os.path.join(self.data_path, "validate_{}".format(self.languages["target"]))
        return utils.prepare_dataset(source_data_file, target_data_file, self.source_converter, self.target_converter, self.GO, self.EOS, self.UNK)

    def get_iterator(self, dataset):
        pad_id = tf.cast(self.PAD, tf.int32)
        dataset = dataset.padded_batch(32, ([None], [None], [None]), (pad_id, pad_id, pad_id), drop_remainder=False)
        return dataset.make_initializable_iterator()

class Inference_Model:
    def __init__(self, data_path, languages, model_path):
        self.model_params = config.MODEL_PARAMS
        self.base_params = config.BASE_PARAMS
        self.data_path = data_path
        self.languages = languages
        self.model_path = model_path
        self.PAD = self.base_params["pad_id"]
        self.GO = self.base_params["go_id"]
        self.EOS = self.base_params["eos_id"]
        self.UNK = self.base_params["unk_id"]
        self.tokenizer = self.get_tokenizer()
        self.detokenizer = self.get_detokenizer()
        self.source_converter = self.get_converter(languages["source"])
        self.target_converter = self.get_converter(languages["target"])
        self.sess = tf.Session()
        # Input
        self.input_holder = tf.placeholder(tf.string, [None, None])
        inputs = self.source_converter.word2id(self.input_holder)
        # Build model
        vocab_size = self.get_vocab_size()
        seq2seq = RNN_Seq2Seq(
        vocab_size["source"], 
        vocab_size["target"], 
        self.model_params["embedding_dim"], 
        self.model_params["cell_type"], 
        self.model_params["rnn_dim"],
        self.model_params["encoder_rnn_layer_num"],
        self.model_params["bidirectional"],
        self.model_params["decoder_rnn_layer_num"],
        self.model_params["attention_dim"],
        self.base_params["go_id"],
        self.base_params["eos_id"],
        self.base_params["pad_id"]
        )
#        output_dict = seq2seq(inputs)
        output_dict = seq2seq(inputs, length_penalty_weight=self.base_params["length_penalty_weight"], coverage_penalty_weight=self.base_params["coverage_penalty_weight"], beam_size=self.base_params["beam_size"])
        fed_inputs = output_dict["fed_inputs"]
        self.attention_weights = output_dict["attention_weights"]
        outputs = output_dict["outputs"]
        self.outputs = self.target_converter.id2word(outputs)
        self.fed_inputs = self.target_converter.id2word(fed_inputs)
        # Save and restore
        self.saver = tf.train.Saver()
        self.sess.run(tf.tables_initializer())
        self.restore_params()

    def __call__(self, inputs): 
        tokenized_inputs = self.process_inputs(inputs)
#        print(tokenized_inputs)
        outputs, fed_inputs, attention_weights = self.sess.run((self.outputs, self.fed_inputs, self.attention_weights), feed_dict={self.input_holder:tokenized_inputs})
#        self.check_inference(id_inputs, id_outputs)
        outputs, tokenized_outputs = self.process_output(outputs)
#        _, tokenized_fed_inputs = self.process_output(fed_inputs)
#        print(tokenized_fed_inputs)
        return outputs, tokenized_inputs, tokenized_outputs, attention_weights

    def check_inference(self, id_inputs, id_outputs):
        print("ID inputs")
        print(id_inputs)
        print("ID outputs")
        print(id_outputs)

    def get_tokenizer(self):
        if self.languages["source"] in ["en"]:
            return utils.English_Tokenizer()
        if self.languages["source"] == "ja":
            return utils.Japanese_Tokenizer()

    def get_detokenizer(self):
        if self.languages["target"] in ["en"]:
            return utils.English_Detokenizer()
        if self.languages["target"] == "ja":
            return utils.Japanese_Detokenizer()
        
    def get_converter(self, language):
        return utils.Word_Id_Converter(os.path.join(self.data_path, "vocab_{}".format(language)), self.UNK)

    def get_vocab_size(self):
        vocab_size = {}
        for key, value in self.languages.items():
            vocab_size[key] = utils.get_file_size(os.path.join(self.data_path, "vocab_{}".format(value)))
        return vocab_size

    def process_inputs(self, inputs): # list of text
        tokens_list = self.tokenizer(inputs)
        return tokens_list

    def process_output(self, outputs):
        outputs = outputs.tolist()
        tokenized_outputs = [list(map(lambda byte: byte.decode("utf-8"), seq)) for seq in outputs]
        outputs = self.detokenizer(tokenized_outputs)
        outputs = [output[:output.find(" <EOS>")] for output in outputs]
        return outputs, tokenized_outputs # list of text

    def restore_params(self):
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring trained parameters...")
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("No saved parameters found")
