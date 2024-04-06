import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # disable tensorflow from showing all debugging logs
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

#device_name = "philipshue"
device_name = "philipshue"

# Load saved model weights and context
context = np.load('./trained_model/{device}/word-context.npy'.format(device=device_name), allow_pickle=True).item()
max_encoder_seq_length = context['encoder_max_seq_length']
max_decoder_seq_length = context['decoder_max_seq_length']
num_encoder_tokens = context['num_encoder_tokens']
num_decoder_tokens = context['num_decoder_tokens']

# Load saved word index mappings
input_word2idx = np.load('./trained_model/{device}/word-input-word2idx.npy'.format(device=device_name), allow_pickle=True).item()
input_idx2word = np.load('./trained_model/{device}/word-input-idx2word.npy'.format(device=device_name), allow_pickle=True).item()
target_word2idx = np.load('./trained_model/{device}/word-target-word2idx.npy'.format(device=device_name), allow_pickle=True).item()
target_idx2word = np.load('./trained_model/{device}/word-target-idx2word.npy'.format(device=device_name), allow_pickle=True).item()

encoder_model = load_model('./trained_model/{device}/encoder-weights.h5'.format(device=device_name), compile=False)
decoder_model = load_model('./trained_model/{device}/decoder-weights.h5'.format(device=device_name), compile=False)

sw_nltk = stopwords.words('english')
sw_nltk.remove('on')
sw_nltk.remove('off')
sw_nltk.remove('is')
stop_words = list(set(sw_nltk + ["want", "could", "please", "hi", "hello", "?", ",", "light"]))

# Decode based on text input
def sentence_to_code(input_sentence):
    input_seq = []
    input_wids = []
    input_sentence = input_sentence.lower().split()
    input_tokens = [tok for tok in input_sentence if tok not in stop_words]
    for word in input_tokens:
        idx = 1  # default idx 1 for <UNK>
        if word in input_word2idx:
            idx = input_word2idx[word]
        input_wids.append(idx)
    input_seq.append(input_wids)

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
        #print("output tokens: ", sample_word)

        target_text_len += 1

        if sample_word != '<SOS>' and sample_word != '<EOS>':
                target_text += '' + sample_word
        
        if sample_word == '<EOS>' or target_text_len >= max_decoder_seq_length:
            terminated = True 
        
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sample_token_idx] = 1    
    return target_text.strip('.')

        
def example_decoding():
    print("=================Examples=================")        
    print("please turn on the philips hue", "---", sentence_to_code("please turn on the philips hue"))
    print("=============================")
    print("I really want to turn off my philips hue", "---",sentence_to_code("I really want to turn off my philips hue"))
    print("=============================")
    print("change the hue light colour to purple", "---",sentence_to_code("change the hue light colour to purple"))
    print("=============================")
    print("switch on philips hue", "---",sentence_to_code("power on hue"))
    print("=============================")
    print("deactivate philips hue", "---",sentence_to_code("deactivate philips hue"))
    print("=============================")
    print("change the philips hue colour to yellow", "---",sentence_to_code("change the philips hue colour to yellow"))
    print("=============================")
    print("I want the philips hue colour to be green", "---", sentence_to_code("I want the philips hue colour to be green"))
    print("=============================")
    print("decrease hue to thirty percent of the brightness", "---", sentence_to_code("decrease hue to thirty percent of the brightness"))
    print("=============================")
    print("change philips hue colour to blue", "---", sentence_to_code("change philips hue colour to blue"))
    print("=============================")
    print("dim philips hue to half of the brightness", "---", sentence_to_code("dim philips hue to half of the brightness"))
    print("=============================")
    print("dim philips hue to quater percent of the brightness", "---", sentence_to_code("dim philips hue to quater percent of the brightness"))
    print("=============================")
    print("dim philips hue to 50 percent of the brightness", "---", sentence_to_code("decrease hue to fifty percent"))
    print("=============================")


def decode(sentence):
    inp_vocab = list(input_word2idx.keys())
    inp_vocab.remove('<PAD>')
    inp_vocab.remove('<UNK>')
    inp = [w.lower() for w in sentence.split() if w not in stop_words]
    #print("inp:", inp)
    #example_decoding()
    if len(set(inp) & set(inp_vocab)) == 0:
        print("Sorry, I don't understand you, please try again")
    else:
        code = sentence_to_code(sentence)
        print(f"input command is: {sentence} \n decoded code is: {code}")

if __name__ == '__main__':
    print("Some examples of command --- code")
    #example_decoding()
    decode("dim philips hue to thirty percent of the brightness")
    decode("change color of philips hue to red")
    decode("switch on philips hue")
    decode("turn off philips hue")
    decode("deactivate philips hue")
    decode("power down philips hue")


    #decode("when button 2 is pressed in streamdeck, switch on philips hue")
    #decode("when button 5 is pressed in streamdeck, turn off philips hue")

    #decode("hello how are you?")