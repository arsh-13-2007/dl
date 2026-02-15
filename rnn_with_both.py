import pandas as pd 
import numpy as np 
from keras.preprocessing.text import Tokenizer
from keras.datasets import imdb
from keras.utils import pad_sequences  # use to do padding 
from keras.models import Sequential
from keras.layers import Dense , Dropout, Flatten , Embedding , SimpleRNN

# (X_train , y_train),(X_test , y_test) = imdb.load_data(num_words=10000)  # this data is already preprocessed not need to convert text into vector 

# print(X_train.shape)

# # first step make all with same length 
# # so  do padding 

# X_train = pad_sequences(X_train , padding='post' , maxlen=100)
# X_test = pad_sequences(X_test , padding='post' , maxlen=100)

# print(X_train.shape)

# model = Sequential()
# model.add( SimpleRNN(32 , input_shape=(100,1) , return_sequences=False))
# model.add( Dense(1 , activation='sigmoid'))

# print( model.summary())

# model.compile(loss='binary_crossentropy' , optimizer='adam' , metrics=['accuracy'])
# model.fit(X_train , y_train, epochs=10 , validation_data=(X_test, y_test))

""" generally we use embedding techinque in rnn """



(X_train , y_train),(X_test , y_test) = imdb.load_data(num_words=10000)  # this data is already preprocessed not need to convert text into vector 

print(X_train.shape)

# first step make all with same length 
# so  do padding 

X_train = pad_sequences(X_train , padding='post' , maxlen=100)
X_test = pad_sequences(X_test , padding='post' , maxlen=100)

print(X_train.shape)

model = Sequential()
model.add(Embedding(input_dim = 10000 ,output_dim = 2 ,input_length=  100 ))
model.add(SimpleRNN(32,return_sequences=False))
model.add( Dense(1 , activation='sigmoid'))

print( model.summary())

model.compile(loss='binary_crossentropy' , optimizer='adam' , metrics=['accuracy'])
model.fit(X_train , y_train, epochs=10 , validation_data=(X_test, y_test))