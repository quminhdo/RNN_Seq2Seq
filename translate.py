import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
from argparse import ArgumentParser
import models
import utils
import config

LANG_DICT = {"en": "English", "ja": "Japanese"}
def change_font(text_list, font):
    for text in text_list:
        text.set_font_properties(font)

def plot_alignments(source_words_list, target_words_list, alignments):
    font = font_manager.FontProperties(fname="/home/minh/je.translator/IPAfont00303/ipag.ttf")
    for i in range(len(source_words_list)):
        source_words = source_words_list[i]
        target_words = target_words_list[i]
        alignment = alignments[i]
        fig, ax = plt.subplots()
        im = ax.imshow(alignment)
        ax.set_xticks(np.arange(len(source_words)))
        ax.set_yticks(np.arange(len(target_words)))
        xlabels = ax.set_xticklabels(source_words)
        ylabels = ax.set_yticklabels(target_words)
        change_font(xlabels, font)
        change_font(ylabels, font)
        ax.set_title("Alignment")
        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", "-d", help="Path of the directory containing data")
    parser.add_argument("--model", "-m", help="aka model_number. Parameters are saved to or restored from a directory named 'source_language'-'target_language'.'model_number'")
    parser.add_argument("--language", "-l", help="format: 'source_language'-'target_language', use only the first two letters")
    args = parser.parse_args()
    data = args.data
    model = args.model
    languages = {"source":args.language[:2], "target":args.language[-2:]}
    model_path = "{}.{}_{}.{}".format(data, languages["source"], languages["target"], model)
    translate = models.Inference_Model(data, languages, model_path)
    print("{} -> {} translator".format(LANG_DICT[languages["source"]], LANG_DICT[languages["target"]]))
    print("Press Ctrl-C to terminate")
    while True:
        try:
            inputs = input("Input {} text: ".format(LANG_DICT[languages["source"]]))
            if inputs == "":
                print("Invalid input. Please try again")
                continue
            outputs, tokenized_inputs, tokenized_outputs, alignments = translate([inputs])
            print("{} translation:".format(LANG_DICT[languages["target"]]), ' '.join(outputs))
            plot_alignments(tokenized_inputs, tokenized_outputs, alignments)
        except KeyboardInterrupt:
            print("\nSee you later.")
            break;

