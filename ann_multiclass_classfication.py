import pandas as pd 
import numpy as  np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow
import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Flatten


(X_train  , y_train) , ( X_test , y_test) = keras.datasets.mnist.load_data()
print( X_train.shape)
plt.imshow(X_train[2])
plt.show()

# ss = StandardScaler()     # kya ham values pe standard scaler use nhi kar skte   doubt

X_test = X_test/255
X_train = X_train/255
# print(X_test)
model = Sequential()

model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128 ,activation ='relu'))
model.add(Dense(32 ,activation ='relu'))
model.add(Dense( 10 , activation = 'softmax'))
model.compile(loss='sparse_categorical_crossentropy' , optimizer='Adam' , metrics=['accuracy'] )
print(model.summary())
history = model.fit(X_train , y_train , epochs=15 , validation_split=0.2)
y_probability=model.predict( X_test)
y_pred = y_probability.argmax(axis = 1 )
accuracy = accuracy_score(y_test , y_pred)
print("accruacy :" ,accuracy)

plt.plot( history.history['accuracy'])
plt.plot( history.history['val_accuracy'])
plt.show() 
