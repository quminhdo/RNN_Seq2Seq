import os
import sys
import shutil
import tensorflow as tf
from argparse import ArgumentParser
import models

class Boolean_Answer:
    YES = ['y', 'Y', "Yes", "yes"]
    NO = ['n', 'N', "No", "no"]

def copy_file(src_path, dst_path):
    with open(src_path) as fin:
        with open(dst_path, 'w') as fout:
            fout.write(fin.read())

def make_model_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        copy_file("config.py", os.path.join(dir_path, "config.py"))
    else:
        if input("Directory {} already exists, do you want to overwrite it? (y/n)".format(dir_path)) in Boolean_Answer.YES:
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)
            copy_file("config.py", os.path.join(dir_path, "config.py"))
        elif input("Continue training with saved parameters? (y/n)") in Boolean_Answer.NO:
            sys.exit(0)
    return dir_path
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", "-d", help="Path of the directory containing data")
    parser.add_argument("--language", "-l", help="format: 'source_language-target_language', use only the first two letters")
    parser.add_argument("--model", "-m", help="aka model_number. Parameters are saved to or restored from a directory named 'source_language'-'target_language'.'model_number'")
    args = parser.parse_args()
    data = args.data
    languages = {"source":args.language[:2], "target":args.language[-2:]}
    model = args.model
    model_path = make_model_dir("{}.{}_{}.{}".format(data, languages["source"], languages["target"], model))
    train_graph = tf.Graph()
    validate_graph = tf.Graph()
    with train_graph.as_default():
        train = models.Training_Model(data, languages, model_path, train_graph)
    with validate_graph.as_default():
        validate = models.Validating_Model(data, languages, model_path, validate_graph)
    log_file = open(os.path.join(model_path, "log"), 'a')
    validate_steps = 1
    while True:
        try:
            train_log, epoch = train()
            print(train_log)
            log_file.write(train_log)
            if epoch % validate_steps == 0:  
                validate_log = validate()
                print(validate_log)
                log_file.write(validate_log)
        except KeyboardInterrupt:
            log_file.close()
            train.sess.close()
            validate.sess.close()
            break;
