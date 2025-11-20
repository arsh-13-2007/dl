import pandas as pd 
import numpy as  np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import accuracy_score , r2_score
import tensorflow
import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense  
from tensorflow.keras.layers import Dropout  
from tensorflow.keras.optimizers import Adam  
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("Admission_Predict_Ver1.1.csv")
print( df.head())

print(df.isnull().sum())

X= df.iloc[:, 1:-1  ]
y = df.iloc[:, -1] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print( X_train , X_test)
mns = MinMaxScaler()
X_scaled_train= mns.fit_transform( X_train )  
X_scaled_test= mns.transform( X_test ) 

model = Sequential()
model.add(Dense(64 , input_dim=7 , activation='relu' , kernel_initializer='glorot_normal'))
# model.add(Dropout(0.2))
model.add(Dense(32 , activation='relu', kernel_initializer='glorot_normal'))
# model.add(Dropout(0.2))
model.add(Dense(1 , activation='linear'))
print(model.summary())


callback = EarlyStopping(
    monitor='val_loss',
    patience=10, 
    verbose=0,
    mode='auto',
    restore_best_weights=False
)
model.compile(loss='mse' , optimizer='adam' , metrics=['accuracy'])
history = model.fit(X_scaled_train, y_train, epochs=200, validation_split=0.2, callbacks=callback)
y_pred = model.predict(X_scaled_test)
accuracy = r2_score(y_test, y_pred)
print("r2_score:", accuracy)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()
