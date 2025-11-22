import tensorflow 
from tensorflow.keras import datasets , layers , models 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
import kerastuner as kt 
import matplotlib.pyplot as plt
import numpy as np

(X_train , y_train ) , ( X_test  , y_test ) = datasets.cifar10.load_data()
print(X_train.shape)
print( X_test.shape)

# print(X_train[0].max())

# preprocessing  

X_train = X_train / 255.0
X_test = X_test / 255.0

# plt.imshow(X_train[0])
# plt.axis('off')
# plt.show()


def build_model(hp):
  model = Sequential()
  counter = 0 
  for i in range(hp.Int('num_layers' , min_value=1 , max_value=5)):
    if counter == 0:
      model.add(Conv2D(
        hp.Int('filter', min_value=64, max_value=128) ,
        activation=hp.Choice('activation'  + str(i) , values = ['relu' , 'sigmoid' , 'tanh']),
        padding=hp.choice('padding'  + str(i), values =['same' , 'valid']) , 
        kernel_size = (3,3) , 
        input_shape=(28 , 28 , 1 )))
      model.add(MaxPooling2D(pool_size=(2,2) , 
        padding=hp.choice('padding'  + str(i), values =['same' , 'valid']) ,
        strides = hp.choice('strides' + str(i) , values=[1,2])))
    else:
      model.add(Conv2D(
        hp.Int('filter', min_value=64, max_value=128) ,
        activation=hp.Choice('activation'  + str(i), values = ['relu' , 'sigmoid' , 'tanh']),
        padding=hp.choice('padding' + str(i) , values =['same' , 'valid']) , 
        kernel_size = (3,3)
        ))
      model.add(MaxPooling2D(pool_size=(2,2) , 
        padding=hp.choice('padding' + str(i) , values =['same' , 'valid']) ,
        strides = hp.choice('strides' + str(i) , values=[1,2])))
    counter +=1 
  model.add(Flatten())
  for i in range(hp.Int('num_layers' , min_value=1 , max_value=10)):
    model.add(Dense(
        units = hp.Int('neurons' + str(i) , min_value= 64 , max_value= 128) , 
        activation=hp.Choice('activation' + str(i) , values = ['relu' , 'sigmoid' , 'tanh']),
    ))
    model.add(Dropout(hp.Choice('dropout'+str(i), values=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])))
  model.add(Dense(10, activation='softmax'))

  model.compile(optimizer= hp.Choice('optimizer' , values=['rmsprop' , 'adam' , 'sgd' ,'nadam' ,'adadelta']) ,
                   loss='sparse_categorical_crossentropy', metrics=['accuracy'])
      

      