from data_utils import load_clean_sentences
from translation_model_utils import create_tokenizer, max_length, encode_output, encode_sequence, build_model, create_lists
from keras.optimizers import RMSprop
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, Sequential
import pandas as pd
import keras


train = load_clean_sentences('/home/pszmelcz/Desktop/translator/code_files/eng-pol_data_train.pkl') #loading train dataset
test = load_clean_sentences('/home/pszmelcz/Desktop/translator/code_files/eng-pol_data_test.pkl')   #loading test dataset

eng_tokenizer = create_tokenizer(train[:, 0])   #creating tokenizer for english words
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(train[:, 0])

pol_tokenizer = create_tokenizer(train[:, 1])   #creating tokenizer for polish words
pol_vocab_size = len(pol_tokenizer.word_index) + 1
pol_length = max_length(train[:, 1])

with open('words.txt', 'w') as f:
   for word in pol_tokenizer.word_index:
       f.write(word + '\n')

print('English Vocabulary Size: {}'.format(eng_vocab_size))
print('English Max Length: {}'.format((eng_length)))

print('Polish Vocabulary Size: {}'.format(pol_vocab_size))
print('Polish Max Length: {}'.format((pol_length)))

trainX = encode_sequence(pol_tokenizer, pol_length, train[:, 1])    #tokezation and pad sequences
trainY = encode_sequence(eng_tokenizer, eng_length, train[:, 0])    
trainY = encode_output(trainY, eng_vocab_size)              #encoding output


testX = encode_sequence(pol_tokenizer, pol_length, train[:, 1])
testY = encode_sequence(eng_tokenizer, eng_length, train[:, 0])
testY = encode_output(testY, eng_vocab_size)


model = build_model(pol_vocab_size, eng_vocab_size, pol_length, eng_length, 512)
#model = load_model('model.h5') 
rms = RMSprop(lr=0.001)
model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')

filename = 'model2.h5'


checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1), epochs=25, batch_size=64, validation_split=0.2, callbacks=[checkpoint], verbose=1)

#model = keras.models.load_model('model.h5')

#preds = model.predict_classes(testX.reshape((testX.shape[0],testX.shape[1])), batch_size=16)

