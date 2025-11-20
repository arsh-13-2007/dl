import pandas as pd 
import numpy as  np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import accuracy_score , r2_score
import tensorflow
import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense  
from tensorflow.keras.layers import Dropout  
from tensorflow.keras.optimizers import Adam  
from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizers import Adam

X_train = np.linspace(-1 ,1 , 20)

y_train = np.array([-0.6561 , -0.3099 , -0.59035, -0.50855, -0.285  , 
                    -0.2443 , -0.02445,  0.00135, -0.2006 ,  0.07475, 
                    -0.1422 ,  0.06515,  0.15265,  0.3521 ,  0.28415,  
                    0.5524 ,  0.23115,  0.20835, 0.4211,  0.60485])
X_test = np.linspace(-1, 1, 20)
y_test = np.array([-0.69415, -0.451  , -0.43005, -0.4484 , -0.1475 ,
                   -0.5019 , -0.28055,  0.24595, -0.21425, -0.0286 ,  
                   0.23415,  0.46575, 0.07955,  0.1973 ,  0.0719 ,
                   0.3639 ,  0.5536 ,  0.3365 , 0.50705,  0.33435])

plt.scatter(X_train , y_train ,c='black', label='train')
plt.scatter(X_test, y_test , c='red' , label='test')
plt.legend()
plt.show()
model = Sequential()
model.add(Dense(128, input_dim=1 , activation='relu'  )) 
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation="linear"))


model.compile(loss='mse' , optimizer='adam' , metrics=['mse'])
callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=2,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False
)
history = model.fit(X_train , y_train , epochs = 100  , validation_split=0.2 , callbacks = callback )
y_pred = model.predict(X_test)
accuracy = r2_score(y_test , y_pred)
print(accuracy)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

plt.scatter(X_train , y_train , c='black' , label='train')
plt.scatter(X_test , y_test , c='red' , label='test')
plt.plot(X_test , y_pred)
plt.legend()
plt.ylim((-1.5 , 1.5)) # So your plot will only show points between –1.5 and 1.5 on the vertical axis
plt.show()