from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import os
from tensorflow.keras import backend
backend.clear_session()
from models import *
from utils import *
import os
import pandas as pd
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # disable tensorflow debugging logs
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding,dot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from nltk.corpus import stopwords
from main import *

""" device_name= "philipshue"
SRC_PATH = '../corpus_generation/primitives/{device}/data/text.txt'.format(device=device_name)
TRG_PATH = '../corpus_generation/primitives/{device}/data/code.txt'.format(device=device_name)
WORD_WEIGHT_PATH = './trained_model/{device}/word-weights.h5'.format(device=device_name) """


CORPUS_PATH = "./sd_hue/corpus.tsv"
df = pd.read_csv(CORPUS_PATH, header=None,sep='\t')
src = df[0].to_list()
trg = df[1].to_list()

""" SRC_PATH = '../corpus_generation/nl.txt'
TRG_PATH = '../corpus_generation/blast.txt'  """
WORD_WEIGHT_PATH = './trained_model/sd_hue/word-weights.h5'

# Define hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 100
HIDDEN_UNITS = 128
MAX_INPUT_SEQ_LEN = 10
MAX_OUTPUT_SEQ_LEN = 35
MAX_VOCAB_SIZE = 100
LERANING_RATE = 0.008
#sampling_probability = 0.5

""" Path of paired training files: 
nl.txt is the natural language command corpus, 
blast.txt is the corresponding executable BLAST code corpus.
WORD_WEIGHT_PATH to save the trained model weights
"""
""" # Loading data
with open(SRC_PATH, 'r', encoding='utf8') as f:
    src = f.read().split('\n')
    
with open(TRG_PATH, 'r', encoding='utf8') as f:
    trg = f.read().split('\n') """

# customize stopwords
sw_nltk = stopwords.words('english')
sw_nltk.remove('on')
sw_nltk.remove('off')
sw_nltk.remove('is')
stop_words = list(set(sw_nltk + ["want", "could", "light", "lamp"]))


# lowercase and remove stop words
def preprocessing(line):
    line_list = [w.lower() for w in line.split() if w.lower() not in stop_words]
    return line_list


# Count frequency of source text and target text
source_counter = Counter()
target_counter = Counter()
source_texts = []
target_texts = []


# read from src to list
prev_words = []
for line in src:
    next_words = preprocessing(line)
    if len(next_words) > MAX_OUTPUT_SEQ_LEN:
        next_words = next_words[0:MAX_OUTPUT_SEQ_LEN]

    if len(prev_words) > 0:
        source_texts.append(prev_words)
        for w in prev_words:
            source_counter[w] += 1

    prev_words = next_words


# Read from trg to list
prev_words = []
for line in trg:
    #tokens = [w for w in line.split()]
    tokens = code_tokenizer(line)
    next_words = [w for w in tokens]
    if len(next_words) > MAX_OUTPUT_SEQ_LEN:
        next_words = next_words[0:MAX_OUTPUT_SEQ_LEN]

    if len(prev_words) > 0:
        target_words = next_words[:]
        target_words.insert(0, '<SOS>')
        target_words.append('<EOS>')
        for w in target_words:
            target_counter[w] += 1
        target_texts.append(target_words)

    prev_words = next_words

""" print("*****************************************")
print("len of target texts: ", len(target_texts))
print("target_counter: ", target_counter)
print("*****************************************") """


# Build word2idx and idx2word dictionary for source texts
source_word2idx = {}
source_word2idx['<PAD>'] = 0
source_word2idx['<UNK>'] = 1 
for idx, word in enumerate(source_counter.most_common(MAX_VOCAB_SIZE)):
    source_word2idx[word[0]] = idx + 2  # 0 and 1 are taken by pad and unk
input_idx2word = dict([(idx, word) for word, idx in source_word2idx.items()])


# Build word2idx and idx2word dictionary for target texts
target_word2idx = {}
target_word2idx['<UNK>'] = 0
for idx, word in enumerate(target_counter.most_common(MAX_VOCAB_SIZE)):
    target_word2idx[word[0]] = idx + 1  # 0 is taken by unk
target_idx2word = dict([(idx, word) for word, idx in target_word2idx.items()])


# Unique word token in souece and target
num_encoder_tokens = len(input_idx2word)
num_decoder_tokens = len(target_idx2word)
""" print("number of unique word token in source: ", num_encoder_tokens)
print("number of unique word token in target: ", num_decoder_tokens) """

#saved_path = "./trained_model/{device}".format(device=device_name)
saved_path = "./trained_model/sd_hue"

isExist = os.path.exists(saved_path)
if not isExist:
    os.makedirs(saved_path)

np.save('{saved_path}/word-input-word2idx.npy'
        .format(saved_path=saved_path), source_word2idx)
np.save('{saved_path}/word-input-idx2word.npy'
        .format(saved_path=saved_path), input_idx2word)
np.save('{saved_path}/word-target-word2idx.npy'
        .format(saved_path=saved_path), target_word2idx)
np.save('{saved_path}/word-target-idx2word.npy'
        .format(saved_path=saved_path), target_idx2word)

#mapping_path = "./trained_model/{device}/mappings".format(device=device_name)
mapping_path = "./trained_model/sd_hue/mappings"
isExist = os.path.exists(mapping_path)

if not isExist:
    os.makedirs(mapping_path)

store_js('{mapping_path}/input-word2idx.js'
         .format(mapping_path=mapping_path), source_word2idx)
store_js('{mapping_path}/input-idx2word.js'
         .format(mapping_path=mapping_path), input_idx2word)
store_js('{mapping_path}/target-word2idx.js'
         .format(mapping_path=mapping_path), target_word2idx)
store_js('{mapping_path}/target-idx2word.js'
         .format(mapping_path=mapping_path), target_idx2word)


encoder_input_data = []
encoder_max_seq_length = 0
decoder_max_seq_length = 0

for input_words, target_words in zip(source_texts, target_texts):
    encoder_input_wids = []
    for w in input_words:
        w2idx = 1  # default [UNK]
        if w in source_word2idx:
            w2idx = source_word2idx[w]
        encoder_input_wids.append(w2idx)

    encoder_input_data.append(encoder_input_wids)
    encoder_max_seq_length = max(len(encoder_input_wids), encoder_max_seq_length)
    decoder_max_seq_length = max(len(target_words), decoder_max_seq_length)
    
""" print("*****************************************")
print("encoder input data top 10", encoder_input_data[:10])
print("encoder max seq len:", encoder_max_seq_length)
print("decoder max seq len:", decoder_max_seq_length)
print("*****************************************") """

context = dict()
context['num_encoder_tokens'] = num_encoder_tokens
context['num_decoder_tokens'] = num_decoder_tokens
context['encoder_max_seq_length'] = encoder_max_seq_length
context['decoder_max_seq_length'] = decoder_max_seq_length
""" print("*****************************************")
print("context:", context)
print("*****************************************") """

np.save('{saved_path}/word-context.npy'.format(saved_path=saved_path), context)
store_js('{saved_path}/mappings/word-context.js'.format(saved_path=saved_path), context)


# batch generator for training
def generate_batch(input_data, output_text_data):
    num_batches = len(input_data) // BATCH_SIZE
    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            encoder_input_data_batch = pad_sequences(input_data[start:end], encoder_max_seq_length)
            decoder_target_data_batch = np.zeros(shape=(BATCH_SIZE, 
                                                        decoder_max_seq_length, 
                                                        num_decoder_tokens))
            decoder_input_data_batch = np.zeros(shape=(BATCH_SIZE, 
                                                       decoder_max_seq_length, 
                                                       num_decoder_tokens))
            for lineIdx, target_words in enumerate(output_text_data[start:end]):
                for idx, w in enumerate(target_words):
                    w2idx = 0  # default [UNK]
                    if w in target_word2idx:
                        w2idx = target_word2idx[w]
                    decoder_input_data_batch[lineIdx, idx, w2idx] = 1
                    if idx > 0:
                        decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
            #print("first batch data example: ", 
            # [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch)
            yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch




X_train, X_test, y_train, y_test = train_test_split(encoder_input_data, target_texts, test_size=0.1)

#print("len of X_train", len(X_train))
#print("len of X_test", len(X_test))

train_gen = generate_batch(X_train, y_train)
test_gen = generate_batch(X_test, y_test)

train_num_batches = len(X_train) // BATCH_SIZE
test_num_batches = len(X_test) // BATCH_SIZE

# Save the best checkpoint after each epoch of training
checkpoint_callback = ModelCheckpoint(filepath=WORD_WEIGHT_PATH,
                             save_freq='epoch',
                             save_best_only=True)



def start_training(use_model):
        # choose between vanilla seq2seq model or seq2seq_attention, or seq2seq_attention_copy
    if use_model == "basicseq2seq":
        print("*************Using basic seq2seq for training**************")
        #global model
        model, encoder_inputs, encoder_states, decoder_lstm, decoder_dense = build_vanilla_model(num_encoder_tokens, 
                                    HIDDEN_UNITS, 
                                    encoder_max_seq_length, 
                                    num_decoder_tokens)
    elif use_model=="attentionseq2seq":
        print("*************Using attention seq2seq for training**************")
        model, encoder_inputs, encoder_states, decoder_lstm, decoder_dense, encoder_outputs, decoder_combined_context = build_attention_model(num_encoder_tokens, 
                                      HIDDEN_UNITS, 
                                      encoder_max_seq_length, 
                                      num_decoder_tokens)
    # TODO:add attention and copy model
    else:
        pass

    optimizer = Adam(learning_rate=LERANING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[ppx])
    #model.compile(optimizer=Adam(), loss=lambda y_true, y_pred: scheduled_sampling_loss(y_true, y_pred, sampling_probability))

    json = model.to_json()
    open('{saved_path}/word-architecture.json'.format(saved_path=saved_path), 'w').write(json)

    model.fit(train_gen, 
        steps_per_epoch=train_num_batches,
        epochs=NUM_EPOCHS,
        verbose=1,
        validation_data=test_gen, 
        validation_steps=test_num_batches, 
        callbacks=[checkpoint_callback])

    encoder_model = Model(encoder_inputs, encoder_states)
    encoder_model.save('{saved_path}/encoder-weights.h5'.format(saved_path=saved_path))


    """ 
    The decoder has different input size during training and inference. 
    During training, it takes target (ground truth) as input but during inference phase there is no target available.
    Thus, the decoder need to be redefined to be saved/used for inference
    """

    #for basicseq2seq
    new_decoder_inputs = Input(batch_shape=(1, None, num_decoder_tokens), name='new_decoder_inputs')
    new_decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='new_decoder_lstm', stateful=True)
    new_decoder_outputs, _, _ = new_decoder_lstm(new_decoder_inputs)
    new_decoder_dense = Dense(units=num_decoder_tokens, activation='softmax', name='new_decoder_dense')
    new_decoder_outputs = new_decoder_dense(new_decoder_outputs)
    new_decoder_lstm.set_weights(decoder_lstm.get_weights())
    new_decoder_dense.set_weights(decoder_dense.get_weights())
    new_decoder_model = Model(new_decoder_inputs, new_decoder_outputs)
    new_decoder_model.save('{saved_path}/decoder-weights.h5'.format(saved_path=saved_path))


start_training("basicseq2seq")