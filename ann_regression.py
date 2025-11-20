import pandas as pd 
import numpy  as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow 
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv("Admission_Predict_Ver1.1.csv")
print(df.head())
print(df.shape)
print(df.isnull().sum())
# print(df.duplicated().sum())

X= df.iloc[:, 1:-1  ]
y = df.iloc[:, -1] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print( X_train , X_test)
mns = MinMaxScaler()
X_scaled_train= mns.fit_transform( X_train )  # it return array 
X_scaled_test= mns.transform( X_test ) # it return array 

model = Sequential()
model.add(Dense(7, activation='relu' , input_dim =7))
model.add(Dense(7, activation='relu' ))
model.add(Dense(1, activation='linear'))

print(model.summary())
model.compile(loss='mean_squared_error',optimizer='Adam' )
history = model.fit( X_scaled_train , y_train , epochs = 100, validation_split=0.2)
y_pred = model.predict(X_scaled_test)
accuracy = r2_score(y_test , y_pred)

print(accuracy)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()





