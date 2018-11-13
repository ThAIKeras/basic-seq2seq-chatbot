import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
import yaml, pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model, load_model
from keras.layers import Input, Embedding, LSTM, CuDNNLSTM, Dense

class InferenceModel(object):
    
    def __init__(self):
        config_name = 'config02'        
        self.__set_config(config_name)
        
        self.__load_tokenizer('models/tokenizer_{}.pkl'.format(config_name))
        
        model = load_model("models/seq2seq-cornell_{}.hdf5".format(config_name))        
        self.__build_inference_model(model)
        
    def __set_config(self, config_name):
        with open('notebooks/{}.yml'.format(config_name)) as config_file:
            configs = yaml.load(config_file)
        
        self.MAX_LEN = configs['params']['sequence']['max_len']
        self.START_TOKEN = configs['params']['sequence']['start_token']
        self.KEPT_SYMBOLS = configs['params']['sequence']['kept_symbols']

        self.EMBEDDING_DIM = configs['params']['model']['embedding_dim']
        self.STATE_DIM = configs['params']['model']['state_dim']
        
    def __process_symbol(self, texts):
        if not self.KEPT_SYMBOLS:
            return texts    
        for s in self.KEPT_SYMBOLS:
            texts = [text.replace(s, ' {} '.format(s)) for text in texts]        
        return texts
        
    def __load_tokenizer(self, file_path):
        with open(file_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
            
        self.word2index = self.tokenizer.word_index
        self.index2word = dict(map(reversed, self.tokenizer.word_index.items()))
        
    def __build_inference_model(self, model):
        """Build an inference model."""
        
        # Get model layers
        embedding_layer = model.get_layer('embedding_layer')
        encoder_rnn = model.get_layer('encoder_rnn')
        decoder_rnn = model.get_layer('decoder_rnn')
        decoder_dense = model.get_layer('decoder_dense')

        encoder_inputs = Input(shape=(None,), name='encoder_inputs')
        decoder_inputs = Input(shape=(None,), name='decoder_inputs')

        # Build an encoder
        x = embedding_layer(encoder_inputs)
        encoder_outputs, state_h, state_c = encoder_rnn(x)
        encoder_states = [state_h, state_c]

        self.encoder_model = Model(encoder_inputs, encoder_states)
        #self.encoder_model.summary()

        # Build a decoder
        decoder_state_input_h = Input(shape=(self.STATE_DIM,))
        decoder_state_input_c = Input(shape=(self.STATE_DIM,))
        decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]

        y = embedding_layer(decoder_inputs)
        y, state_h, state_c = decoder_rnn(y, initial_state=decoder_state_inputs)
        decoder_outputs = decoder_dense(y)
        decoder_state_outputs = [state_h, state_c]

        self.decoder_model = Model(
            [decoder_inputs] + decoder_state_inputs,
            [decoder_outputs] + decoder_state_outputs)

        #self.decoder_model.summary()
        
    def __transform_text(self, text):
        sequences = self.tokenizer.texts_to_sequences(text)
        data = pad_sequences(sequences, maxlen=self.MAX_LEN, padding='post', truncating='post')
        return data

    def __decode_sequence(self, input_seq):
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.array([[self.tokenizer.word_index[self.START_TOKEN]]])

        stop_condition = False
        decoded_sentence = []   

        word_count = 0

        while True:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])

            if (sampled_token_index == 0 or word_count >= self.MAX_LEN):
                break

            decoded_sentence.append(self.index2word[sampled_token_index])
            word_count += 1

            states_value = [h, c]
            target_seq = np.array([[sampled_token_index]])  

        return ' '.join(decoded_sentence)
    
    def __format_text(self, text):
        if not self.KEPT_SYMBOLS:
            return text    
        for s in self.KEPT_SYMBOLS:
            text = text.replace(' {}'.format(s), s)
        return text    
    
    def reply(self, question):
        return self.__format_text(self.__decode_sequence(self.__transform_text(self.__process_symbol([question]))))
    
class DialogueManager(object):
    
    def __init__(self):
        print("Loading resources...")        
        self.inference_model = InferenceModel()
       
    def generate_answer(self, question):
        """Generate a reply by using a trained seq2seq model"""        
        return self.inference_model.reply(question)
    
# Test my implementation
test_questions = ["Hi! How are you today?",
                  "Annie, are you ok?",
                  "What is your name?"]

if __name__ == "__main__":
    manager = DialogueManager()        
    
    for q in test_questions:
        print("Q: {}".format(q))
        print("A: {}".format(manager.generate_answer(q)))
