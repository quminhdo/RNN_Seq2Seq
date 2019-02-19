from collections import defaultdict

MODEL_PARAMS = defaultdict(
    lambda:None,
    embedding_dim = 512,
    cell_type = "lstm",
    rnn_dim = 512,
    encoder_rnn_layer_num = 3,
    bidirectional = False,
    decoder_rnn_layer_num = 3,
    attention_dim = 512
)

TRAIN_PARAMS = defaultdict(
    lambda:None,
    learning_rate = 0.001,
    batch_size = 32,
    decay_rate = 0.5,
    decay_step = 100,
)

IKNOW_DATA_PARAMS = defaultdict(
    lambda : None,
    data_path = "iknow",
    validate_size = 1500,
    vocab_en_threshold = 2,
    vocab_ja_threshold = 2,
)

JESC_DATA_PARAMS = defaultdict(
    lambda : None,
    vocab_en_threshold = 10,
    vocab_ja_threshold = 10,
)

BASE_PARAMS = defaultdict(
    lambda:None,
    pad_id = 0,
    go_id = 1,
    eos_id = 2,
    unk_id = 3,
    tokenize_method = "",
    beam_size=3,
    length_penalty_weight=0.5,
    coverage_penalty_weight=0.5,
)
