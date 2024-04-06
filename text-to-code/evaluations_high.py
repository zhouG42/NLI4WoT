import keras.optimizers
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import os
import shutil
import  time
from utils import *
import os
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, concatenate, dot
from tensorflow.keras.layers import Dot, Activation, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers.legacy import Adam
#from tensorflow.keras.optimizers import AdamW,RMSprop, Adadelta, Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from nltk.corpus import stopwords
import pandas as pd
tf.random.set_seed(42)
import numpy as np
np.random.seed(42)
import keras
keras.utils.set_random_seed(42)

device_name = "philipshue"
#device_name = "twoprimitives"

#device_name="event_property"
#device_name = "allthree"

# Check if the folder exists
saved_path = "./trained_model/{device}/high".format(device=device_name)
isExist = os.path.exists(saved_path)
if isExist:
    shutil.rmtree(saved_path)
else:
    os.makedirs(saved_path)


WORD_WEIGHT_PATH = './trained_model/{device}/high/word-weights.h5'.format(device=device_name)
df = pd.read_csv('../corpus_generation/primitives/{device}/data_high/corpus.txt'.format(device=device_name), header=None,sep='\t')
#df = pd.read_csv('./corpus.tsv'.format(device=device_name), header=None,sep='\t')

src = df[0].to_list()
trg = df[1].to_list()

# Define hyperparameters
BATCH_SIZE = 128
NUM_EPOCHS = 15
HIDDEN_UNITS = 128
MAX_INPUT_SEQ_LEN = 20
MAX_OUTPUT_SEQ_LEN = 25
MAX_VOCAB_SIZE = 100
LERANING_RATE = 0.005

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
stop_words = list(set(sw_nltk + ["want", "can", "light", "smart"]))

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




np.save('./trained_model/{device}/high/word-input-word2idx.npy'.format(device=device_name), source_word2idx)
np.save('./trained_model/{device}/high/word-input-idx2word.npy'.format(device=device_name), input_idx2word)
np.save('./trained_model/{device}/high/word-target-word2idx.npy'.format(device=device_name), target_word2idx)
np.save('./trained_model/{device}/high/word-target-idx2word.npy'.format(device=device_name), target_idx2word)

mapping_path = "./trained_model/{device}/high/mappings".format(device=device_name)
isExist = os.path.exists(mapping_path)
if not isExist:
    os.makedirs(mapping_path)

store_js('./trained_model/{device}/high/mappings/input-word2idx.js'.format(device=device_name), source_word2idx)
store_js('./trained_model/{device}/high/mappings/input-idx2word.js'.format(device=device_name), input_idx2word)
store_js('./trained_model/{device}/high/mappings/target-word2idx.js'.format(device=device_name), target_word2idx)
store_js('./trained_model/{device}/high/mappings/target-idx2word.js'.format(device=device_name), target_idx2word)


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
    
context = dict()
context['num_encoder_tokens'] = num_encoder_tokens
context['num_decoder_tokens'] = num_decoder_tokens
context['encoder_max_seq_length'] = encoder_max_seq_length
context['decoder_max_seq_length'] = decoder_max_seq_length
np.save('./trained_model/{device}/high/word-context.npy'.format(device=device_name), context)
store_js('./trained_model/{device}/high/mappings/word-context.js'.format(device=device_name), context)


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
            yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch




X_train, X_test, y_train, y_test = train_test_split(encoder_input_data, target_texts, test_size=0.2, shuffle=True, random_state=42)


train_gen = generate_batch(X_train, y_train)
test_gen = generate_batch(X_test, y_test)

train_num_batches = len(X_train) // BATCH_SIZE
test_num_batches = len(X_test) // BATCH_SIZE

# Save the best checkpoint after each epoch of training
checkpoint_callback = ModelCheckpoint(filepath=WORD_WEIGHT_PATH, save_freq='epoch', save_best_only=True)


start = time.time()
######################## Basic Seq2seq ########################

encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_dim=num_encoder_tokens, 
                            output_dim=HIDDEN_UNITS,
                            input_length=encoder_max_seq_length)
encoder_lstm = LSTM(units=HIDDEN_UNITS,
                    return_state=True,
                    dropout=0.5, 
                    name="encoder_lstm")
encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
encoder_states = [encoder_state_h, encoder_state_c]

#decoder
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(units=HIDDEN_UNITS,
                    return_sequences=True, 
                    return_state=True,
                    dropout=0.5, 
                    name="decoder_lstm")
decoder_outputs, _ , _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(units=num_decoder_tokens,
                       activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)    
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
print(model.summary()) 

#############################################

###################  Bidirectional Seq2Seq ##########################
"""
decoder_lstm_units = HIDDEN_UNITS * 2
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_dim=num_encoder_tokens, 
                            output_dim=HIDDEN_UNITS,
                            input_length=encoder_max_seq_length)

# Use Bidirectional LSTM instead of regular LSTM
encoder_lstm = Bidirectional(LSTM(units=HIDDEN_UNITS,
                                  return_sequences=True,
                                  return_state=True,
                                  name="encoder_lstm"))

encoder_outputs, forward_state_h, forward_state_c, backward_state_h, backward_state_c = encoder_lstm(
    encoder_embedding(encoder_inputs))
# Concatenate the forward and backward states
encoder_state_h = Concatenate()([forward_state_h, backward_state_h])
encoder_state_c = Concatenate()([forward_state_c, backward_state_c])
encoder_states = [encoder_state_h, encoder_state_c]
#decoder
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(units=decoder_lstm_units,  # Use the updated decoder_lstm_units
                    return_sequences=True, 
                    return_state=True,
                    name="decoder_lstm")
decoder_outputs, _ , _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(units=num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)    
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
print(model.summary()) 
"""
#############################################




# ...

# Attention model based on the base model

"""
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_dim=num_encoder_tokens, 
                              output_dim=HIDDEN_UNITS,
                              input_length=encoder_max_seq_length)
encoder_lstm = LSTM(units=HIDDEN_UNITS,
                    return_sequences=True,
                    return_state=True,
                    name="encoder_lstm")
encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
encoder_states = [encoder_state_h, encoder_state_c]

#decoder
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(units=HIDDEN_UNITS,
                    return_sequences=True, 
                    return_state=True,
                    name="decoder_lstm")
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# Scaled Dot-Product Attention for the decoder
context_vector = scaled_dot_product_attention(decoder_outputs, encoder_outputs, encoder_outputs)

# Concatenate context vector with decoder LSTM outputs
decoder_outputs_with_attention = tf.add(context_vector, decoder_outputs)
# Dense layer to map to the output vocabulary
decoder_dense = Dense(units=num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs_with_attention)

# Create the model with attention mechanism
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
print(model.summary())
"""

#############################################


optimizer = Adam(learning_rate=LERANING_RATE)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', ppx])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)

""" # Define EarlyStopping callback
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Metric to monitor (e.g., validation loss)
    patience=3,           # Number of epochs with no improvement before stopping
    restore_best_weights=True  # Restores model weights to the point of best performance
)
 """
json = model.to_json()
open('./trained_model/{device}/high/word-architecture.json'.format(device=device_name), 'w').write(json)


model.fit(train_gen, 
    steps_per_epoch=train_num_batches,
    epochs=NUM_EPOCHS,
    verbose=1,
    validation_data=test_gen, 
    validation_steps=test_num_batches, 
    callbacks=[tensorboard_callback])

encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.save('./trained_model/{device}/high/encoder-weights.h5'.format(device=device_name))
end = time.time()

print("******** Training time: **********", end-start)

new_decoder_inputs = Input(batch_shape=(1, None, num_decoder_tokens), name='new_decoder_inputs')
new_decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='new_decoder_lstm', stateful=True)
new_decoder_outputs, _, _ = new_decoder_lstm(new_decoder_inputs)
new_decoder_dense = Dense(units=num_decoder_tokens, activation='softmax', name='new_decoder_dense')
new_decoder_outputs = new_decoder_dense(new_decoder_outputs)
new_decoder_lstm.set_weights(decoder_lstm.get_weights())
new_decoder_dense.set_weights(decoder_dense.get_weights())
new_decoder_model = Model(new_decoder_inputs, new_decoder_outputs)
new_decoder_model.save('./trained_model/{device}/high/decoder-weights.h5'.format(device=device_name))

