from cmath import inf
from tensorflow.keras.models import Model, model_from_json, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import argparse
import time


# Load saved model weights and context
context = np.load('./trained_model/philipshue/word-context.npy', allow_pickle=True).item()
max_encoder_seq_length = context['encoder_max_seq_length']
max_decoder_seq_length = context['decoder_max_seq_length']
num_encoder_tokens = context['num_encoder_tokens']
num_decoder_tokens = context['num_decoder_tokens']

# Load saved word index mappings
input_word2idx = np.load('./trained_model/philipshue/word-input-word2idx.npy', allow_pickle=True).item()
input_idx2word = np.load('./trained_model/philipshue/word-input-idx2word.npy', allow_pickle=True).item()
target_word2idx = np.load('./trained_model/philipshue/word-target-word2idx.npy', allow_pickle=True).item()
target_idx2word = np.load('./trained_model/philipshue/word-target-idx2word.npy', allow_pickle=True).item()

encoder_model = load_model('./trained_model/philipshue/encoder-weights.h5', compile=False)
decoder_model = load_model('./trained_model/philipshue/decoder-weights.h5', compile=False)


# Decode based on text input
def sentence_to_code(input_sentence):
    input_seq = []
    input_wids = []
    # stop words not removed for now
    stopwords = ["please", "I", "want", "the", "a", "an", "of", "that", "could", "you", "light", "my", "i"]
    input_sentence = input_sentence.lower().split()
    input_tokens = [tok for tok in input_sentence if tok not in stopwords]
    #print("input_tokens: ", input_tokens)
    for word in input_tokens:
        idx = 1  # default idx 1 for <UNK>
        if word in input_word2idx:
            idx = input_word2idx[word]
        input_wids.append(idx)
    input_seq.append(input_wids)
    #print("input_wids", input_wids)
    #print("input_seq", input_seq)

    # pad to max encoder input length
    input_seq = pad_sequences(input_seq, max_encoder_seq_length)
    #print("input_seq_after_pad", input_seq)
    states_value = encoder_model.predict(input_seq, verbose=0)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_word2idx['<SOS>']] = 1
    target_text = ''
    target_text_len = 0
    terminated = False
    decoder_model.layers[-2].reset_states(states=states_value)

    while not terminated:
        output_tokens = decoder_model.predict(target_seq, verbose=0)
        sample_token_idx = np.argmax(output_tokens[0, -1, :])
        sample_word = target_idx2word[sample_token_idx]
        target_text_len += 1

        if sample_word != '<SOS>' and sample_word != '<EOS>':
                target_text += ' ' + sample_word
        
        if sample_word == '<EOS>' or target_text_len >= max_decoder_seq_length:
            terminated = True 
        
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sample_token_idx] = 1
    
    return target_text.strip('.')


# test set corpus accuracy score
test_text = []
test_code = []
with open('../corpus_generation/primitives/philipshue/data/test.txt', 'r') as file:
    # Read each line in the file
    for line in file:
        # Split the line using the tab character and strip any leading/trailing whitespace
        text, code = line.strip().split('\t')
        
        # Append the values to the respective lists
        test_text.append(text)
        test_code.append(code.replace(" ", ""))

test_size = len(test_text[:100])
print("size of test set", test_size)


def acc():
    decoded = [sentence_to_code(text).replace(" ", "").strip() for text in test_text[:100]]
    # Use zip to pair the decoded results and test_code
    results = zip(decoded, test_code)
    # Calculate the score using list comprehension
    score = sum(1 for decoded, ground_truth in results if decoded == ground_truth)
    # Re-create the results iterator for printing
    results = zip(decoded, test_code)
    
    mismatched_pairs = [(decoded, ground_truth) for decoded, ground_truth in results if decoded != ground_truth]
    
    print("Mismatched pairs:")
    for decoded, ground_truth in mismatched_pairs:
        print(f"Decoded: {decoded}, Ground Truth: {ground_truth}")
    
    print(score)
    acc = score / test_size
    return acc

start_time = time.time()
print(acc())
end_time = time.time()
# Calculate the runtime
runtime = end_time - start_time
print(f"Function executed in {runtime} seconds.")