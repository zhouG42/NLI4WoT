import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # disable tensorflow from showing all debugging logs
import tensorflow as tf
from utils import *
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, Dropout
from tensorflow.keras.layers import Activation, dot, concatenate
from tensorflow.keras.initializers import Constant
from keras.regularizers import l2


def build_vanilla_model(num_encoder_tokens, hidden_units, encoder_max_seq_len, num_decoder_tokens):
        encoder_inputs = Input(shape=(None,))
        encoder_embedding = Embedding(input_dim=num_encoder_tokens, 
                                  output_dim=hidden_units,
                                  input_length=encoder_max_seq_len)
        encoder_lstm = LSTM(units=hidden_units,
                            return_state=True,
                            name="encoder_lstm")
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
        encoder_states = [encoder_state_h, encoder_state_c]
        
        #decoder
        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        decoder_lstm = LSTM(units=hidden_units,
                            return_sequences=True, 
                            return_state=True,
                            name="decoder_lstm")
        decoder_outputs, _ , _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(units=num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)    
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        print(model.summary())
        return model, encoder_inputs, encoder_states, decoder_lstm, decoder_dense

def build_bidirectional_model(num_encoder_tokens, hidden_units, encoder_max_seq_len, num_decoder_tokens):
    # Encoder
    encoder_inputs = Input(shape=(None,))
    encoder_embedding = Embedding(input_dim=num_encoder_tokens, 
                                  output_dim=hidden_units,
                                  input_length=encoder_max_seq_len)
    
    # Replace LSTM with Bidirectional LSTM
    encoder_lstm = Bidirectional(LSTM(units=hidden_units,
                                      return_state=True,
                                      name="encoder_lstm"))
    
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding(encoder_inputs))
    encoder_state_h = concatenate([forward_h, backward_h])
    encoder_state_c = concatenate([forward_c, backward_c])
    encoder_states = [encoder_state_h, encoder_state_c]

    # Decoder
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(units=hidden_units*2,  # Double the units to match the bidirectional encoder
                        return_sequences=True, 
                        return_state=True,
                        name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(units=num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    print(model.summary())
    return model, encoder_inputs, encoder_states, decoder_lstm, decoder_dense




"""
Decoder enhanced with Attention mechanism
"""

def build_attention_model(num_encoder_tokens, hidden_units, encoder_max_seq_len, num_decoder_tokens):
        encoder_inputs = Input(shape=(None,))
        encoder_embedding = Embedding(input_dim=num_encoder_tokens, 
                                  output_dim=hidden_units,
                                  input_length=encoder_max_seq_len)
        encoder_lstm = LSTM(units=hidden_units, 
                            return_state=True,
                            return_sequences=True,
                            name="encoder_lstm")       
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
        encoder_states = [encoder_state_h, encoder_state_c]
        #decoder
        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        decoder_lstm = LSTM(units=hidden_units,  
                            return_sequences=True,
                            return_state=True,
                            name="decoder_lstm")
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        # Attention
        attention = dot([decoder_outputs, encoder_outputs], axes=[2,2])
        attention_weight = Activation('softmax')(attention)
        context = dot([attention_weight, encoder_outputs], axes=[2,1])
        decoder_combined_context = concatenate([context, decoder_outputs])
        decoder_dense = Dense(units=num_decoder_tokens, activation='softmax', name="decoder_dense")
        decoder_outputs = decoder_dense(decoder_combined_context)           
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        print(model.summary())
        return model, encoder_inputs, encoder_states, decoder_lstm, decoder_dense, encoder_outputs, decoder_combined_context




"""
Decoder enhanced with Attention and copy mechanism
"""
# todo