from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import re
import numpy as np
import nltk
import tensorflow as tf
from nltk.stem.porter import PorterStemmer
from tensorflow.keras import backend as K
K.clear_session()
from tensorflow.keras.losses import categorical_crossentropy
import math
from tensorflow.keras.losses import sparse_categorical_crossentropy
import nltk
nltk.download('stopwords')
""" 
Text preprocessing includes but not limited to:
- lowercase all words
- tokenize input sentence
- remove non-meaningful characters
- remove stopwords
- stem/lemmatize word
"""

def basic_tokenizer(text):
    tokens = text.split().lower()
    return tokens

def code_tokenizer(code):
    pattern = r'\w+|\.|,|;|=>|:|\'[^\']*\'|#\w+|\(|\)|{|}|\[|\]'    
    tokens = re.findall(pattern, code)
    return tokens



def nltk_tokenizer(text):
    return nltk.word_tokenize(text)


def text_cleaning(text):
    txt = text.lower()
    txt = re.sub(r'[\(\)\"#\/@;:\'<>\{\}\=~|\.\?]', '', txt)
    return txt

# customized stopwords list
def remove_stopwords(text):
    stopwords = ["please", "I", "want", "the", "a", "an", "of", "that", "could", "you", "light", "my", "i"]
    return [tok for tok in text if tok not in stopwords]
    
def stem(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word)


# Save word-idx mapping and model context to js file, for later tensorflowjs convention
def store_js(filename, data):
    with open(filename, 'w') as f:
        f.write('module.exports = ' + json.dumps(data, indent=2))

# perplexity as optim metric
def ppx(y_true, y_pred):
    loss = categorical_crossentropy(y_true, y_pred, label_smoothing=0.1)
    perplexity = K.cast(K.pow(math.e, K.mean(loss, axis=-1)), K.floatx())
    return perplexity


def scheduled_sampling_loss(y_true, y_pred, sampling_probability):
    # Flatten the true labels
    y_true = K.flatten(y_true)
    
    # Generate a random probability for each element in the batch
    sample_prob = K.random_uniform(K.shape(y_true), 0, 1)
    
    # Use teacher forcing if the random probability is less than the scheduled probability
    use_teacher_forcing = K.less(sample_prob, sampling_probability)
    
    # Create a mask where 1 indicates using teacher forcing and 0 indicates using model predictions
    mask = K.cast(use_teacher_forcing, dtype='float32')
    
    # Compute the loss with either teacher forcing or model predictions based on the mask
    loss = (1 - mask) * categorical_crossentropy(y_true, y_pred) + mask * sparse_categorical_crossentropy(y_true, y_true)
    return loss


## Use pretrained word embedding to initialize encoder embedding layer
import numpy as np
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant

def load_glove_embeddings(embedding_path, word_index, embedding_dim):
    embeddings_index = {}
    with open(embedding_path, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(word_index), embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            if len(embedding_vector) == embedding_dim:
                embedding_matrix[i] = embedding_vector
            else:
                print(f"Warning: Word '{word}' does not have the expected embedding dimension of {embedding_dim}."
                      f" Using random initialization for this word.")
                # Use random initialization for words with incorrect embedding dimensions
                embedding_matrix[i] = np.random.uniform(-0.1, 0.1, embedding_dim)

    return embedding_matrix


#####################  attention ########################
def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = tf.shape(query)[-1]
    
    # Compute attention scores
    scores = tf.matmul(query, key, transpose_b=True)
    scores = scores / tf.math.sqrt(tf.cast(d_k, tf.float32))
    
    # Apply mask (if provided)
    if mask is not None:
        scores = scores + (mask * -1e9)
    
    # Compute attention probabilities
    attention_probs = tf.nn.softmax(scores, axis=-1)
    
    # Calculate context vector
    context_vector = tf.matmul(attention_probs, value)
    
    return context_vector