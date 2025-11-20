import pandas as pd
import numpy as np
import tensorflow as tf 
import kerastuner as kt
from keras.datasets import mnist
from keras import Sequential
from keras.layers import Dense , Flatten , Conv2D

(X_train  , y_train ) , (X_test , y_test) = mnist.load_data()
# X_train = X_train.reshape(-1, 28, 28, 1)
# X_test  = X_test.reshape(-1, 28, 28, 1)

# X_train = X_train / 255.0
# X_test  = X_test / 255.0
#                                          1. padding ='valid' 
#                                        VALID Padding =  (No padding) 
# model = Sequential()

# model.add(Conv2D(32 , kernel_size=(3,3) , padding='valid' , activation='relu' , input_shape=(28,28,1)))
# model.add(Conv2D(32 , kernel_size=(3,3) , padding='valid' , activation='relu' ))
# model.add(Conv2D(32 , kernel_size=(3,3) , padding='valid' , activation='relu' ))


# model.add(Flatten())

# model.add(Dense(128 , activation='relu'))
# model.add( Dense(10 , activation='softmax'))

# print(model.summary())




#                                          1. padding ='same' 
#                                          same Padding = (padding) 
model = Sequential()

model.add(Conv2D(32 , kernel_size=(3,3) , padding='same' , activation='relu' , input_shape=(28,28,1)))
model.add(Conv2D(32 , kernel_size=(3,3) , padding='same' , activation='relu' ))
model.add(Conv2D(32 , kernel_size=(3,3) , padding='same' , activation='relu' ))


model.add(Flatten())

model.add(Dense(128 , activation='relu'))
model.add( Dense(10 , activation='softmax'))

print(model.summary())
