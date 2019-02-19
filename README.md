# RNN_Seq2Seq
### 1. Introduction
Implementation of a Sequence-to-Sequence model using Luong Attention and GNMT's Beam Search.

Language pair: Japanese - English

### 2. How to use
- Prepare a directory (e.g. "data_dir") containing training data (e.g. "train_en", "train_ja"), validation data (e.g. "validate_en", "validate_en") and vocabulary files(e.g. "vocab_ja", "vocab_en").
- To train the model, use command **python3 train.py -d "data_dir" -l "ja-en"/"en-ja" -m "model_number"**. Trained parameters, log file, and config.py are saved in a newly created directory named "data_dir.ja-en.model_number" or "data_dir.en-ja.model_number"
- To translate using trained model, use command **python3 translate -d "data_dir" -l "ja-en"/"en-ja" -m "model_number"**. A heatmap of attention weights is plotted after each translation.
