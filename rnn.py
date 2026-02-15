from keras import Sequential 
from keras.layers import SimpleRNN , Dense 


model = Sequential()
model.add(SimpleRNN(3 , input_shape=(4, 5)))
model.add(Dense(1 , activation='sigmoid'))

print(model.summary())