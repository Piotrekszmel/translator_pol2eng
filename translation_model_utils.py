from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical 
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Dense, RepeatVector, LSTM

def create_lists(l):
    sent_en = []
    sent_pl = []
    for i in range(len(l)):
        sent_en.append(l[i][0])
        sent_pl.append(l[i][1])
    print(len(sent_en), ' ', len(sent_pl))
    return sent_en, sent_pl

def create_tokenizer(lines):
    tokenizer = Tokenizer() 
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(lines):
    return max(len(line.split()) for line in lines)  #max numbers of words in one sequence

def encode_sequence(tokenizer, length, lines):  #encode and pad sequences
    X = tokenizer.texts_to_sequences(lines)
    X = pad_sequences(X, maxlen=length, padding='post')
    return X

def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = np.asarray(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y    
  
def build_model(in_vocab, out_vocab, in_timesteps, out_timesteps, n):
    model = Sequential()
    model.add(Embedding(in_vocab, n , input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(n))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(n, return_sequences=True))
    model.add(Dense(out_vocab, activation='softmax'))
    return model