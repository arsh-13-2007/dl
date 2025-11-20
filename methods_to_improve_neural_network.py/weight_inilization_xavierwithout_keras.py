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
model.add(Dense(64 , input_dim=7 , activation='tanh'))
# model.add(Dropout(0.2))
model.add(Dense(32 , activation='tanh'))
# model.add(Dropout(0.2))
model.add(Dense(1 , activation='linear'))
print(model.summary())

# print( model.get_weights()) 

# xavier method for initallised weights
initial_weights = model.get_weights()
initial_weights[0] = np.random.randn(7, 64)*np.sqrt(1/7) 
initial_weights[1] = np.zeros(model.get_weights()[1].shape)
initial_weights[2] = np.random.randn(64, 32)*np.sqrt(1/64) 
initial_weights[3] = np.zeros(model.get_weights()[3].shape)
initial_weights[4] = np.random.randn(32, 1)*np.sqrt(1/32) 
initial_weights[5] = np.zeros(model.get_weights()[5].shape)

model.set_weights(initial_weights)
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
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()
