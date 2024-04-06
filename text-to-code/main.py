#from inference import *
from models import * 
import os
import argparse
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # disable tensorflow from showing all debugging logs
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)


parser = argparse.ArgumentParser()
parser.add_argument('-corpus', help= 'choose corpus of device')
parser.add_argument('-task', help="task to run: train, test, or inference")
parser.add_argument('-model', help= 'model architecture: basicseq2seq, attentoinseq2seq, and ')

args = parser.parse_args()


if __name__ == '__main__':
    corpus = args.corpus
    if corpus in ["hue","philipshue", "philips hue"]:
        task = args.task
        if task == "train":
            from train import start_training
            device_name = args.corpus
            use_model = args.model
            print("Starting training with {use_model}, trained model will be saved in /trained_models after training finishes".format(use_model=use_model))
            start_training(use_model)

        elif task == "test":
            #text=input("which evaluation metrics?")
            from test import accuracy
            score = accuracy()
            print("Accuracy score is: ", score)

        elif task == "inference":
            #text = input("Please enter a command to inference:")
            #inference(text)
            example_decoding()
    else:
        print("**********To be implemented**********")
