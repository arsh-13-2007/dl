import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import tensorflow as tf 
import kerastuner as kt 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.datasets import mnist
from keras.layers import Dense , Flatten , Conv2D , MaxPooling2D


(X_train  , y_train ) , (X_test , y_test) = mnist.load_data()


model = Sequential()
model.add(Conv2D(32 ,kernel_size=(3,3) ,  padding= 'same' , activation='relu' , input_shape=(28 , 28, 1)))
model.add(MaxPooling2D(padding='valid' , strides=2  ,pool_size=(2,2)))
model.add(Conv2D(32 ,kernel_size=(3,3) ,  padding= 'same' , activation='relu' ))
model.add(MaxPooling2D(padding='valid' , strides=2  ,pool_size=(2,2)))


model.add( Flatten())

model.add(Dense(128 , activation='relu'))
model.add(Dense(1 , activation='softmax'))

print(model.summary())