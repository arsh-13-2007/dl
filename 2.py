import numpy as np 
import pandas as pd  
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score , confusion_matrix 
import tensorflow 

from tensorflow.keras import datasets , layers , models 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten , BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
import kerastuner as kt 
import matplotlib.pyplot as plt
import numpy as np
ss = StandardScaler()
df = pd.read_csv('diabetes.csv')

print(df.isnull().sum())
print(df.describe())
print(df.info())


X=df.iloc[:,:-1 ].values
y=df.iloc[:,-1].values
X =ss.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = Sequential()
# model.add(Dense(128 , activation = 'relu' , input_dim=8  ))
# model.add(Dropout(0.4))

# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.4))
# # model.add(BatchNormalization())
# model.add(Dense(1 , activation='sigmoid'))

# model.compile(optimizer='adam' , loss='binary_crossentropy' , metrics=['accuracy'])

# model.fit(X_train , y_train , batch_size=32 , epochs=100,validation_data=(X_test, y_test))




# hyper parameter tuning in deep learning 

def build_model(hp):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=8))
    for i in range(hp.Int('num_layers' , min_value=1 , max_value=10)):
        units= hp.Int('units' +str(i) ,min_value= 32 ,  max_value=128 )
        model.add(Dense(units= units ,
                        activation=hp.Choice('activation' +str(i) , values=['sigmoid', 'elu' , 'tanh' , 'selu' , 'relu'])))
        model.add(Dropout(hp.Choice('dropout' +str(i) , values=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])))
        model.add(BatchNormalization())
    model.add(Dense(1, activation= 'sigmoid'))

    optimizers = hp.Choice('optimizer' ,values=['adam' , 'sgd' , 'rmsprop' , 'adadelta'])
    model.compile(optimizer=optimizers , loss='binary_crossentropy' , metrics=['accuracy'])

    return model 


tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    directory='my_dir',
    project_name='diabetes_tuning',
    overwrite=True 
)


tuner.search(X_train , y_train , epochs=5, validation_data=(X_test , y_test))

print(tuner.get_best_hyperparameters()[0].values)

model = tuner.get_best_models(num_models=1)[0]

model.summary()

model.fit(X_train, y_train , batch_size=32 , epochs=200 , validation_data=(X_test , y_test))
