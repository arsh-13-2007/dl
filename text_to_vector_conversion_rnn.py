import numpy as np
import pandas as pd 
from  keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences   # use to padding in text 
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

tokenizer = Tokenizer(oov_token='<nothing>')
tokenizer.fit_on_texts(docs)
print(tokenizer.word_index)
print(tokenizer.word_counts)
print( tokenizer.document_count) # output = number of sentence in our data 

sequences = tokenizer.texts_to_sequences(docs)  # import step 
print(sequences)  # make sequence 

sequences=pad_sequences(sequences, padding='post')  # use to do padding  we can do post or pre also it show that where we want to add zero phle or baad mein 
print(sequences)

