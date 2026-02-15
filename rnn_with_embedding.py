import numpy as np 
import pandas as pd 
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense , Dropout , SimpleRNN , Embedding
docs = ['go india',
		'india india',
		'hip hip hurray',
		'jeetega bhai jeetega india jeetega',
		'bharat mata ki jai',
		'kohli kohli',
		'sachin sachin',
		'dhoni dhoni',
		'modi ji ki jai',
		'inquilab zindabad']

tokenizer = Tokenizer(oov_token='<nothing')
tokenizer.fit_on_texts(docs) # it step give number to each step 
print(tokenizer.word_index)
print(len(tokenizer.word_index))
print(tokenizer.word_counts)
print( tokenizer.document_count)  # this show number of sentence in our data 
sequence = tokenizer.texts_to_sequences(docs)
print(sequence)

sequence= pad_sequences(sequence ,padding='post')
print(sequence)


model = Sequential()
model.add(Embedding(17 , output_dim=2 , input_length=5))
# output_dim , input_length are hyper parameter we can do  tuning using keras tuner
print( model.summary())
# OUTPUT_DIM 