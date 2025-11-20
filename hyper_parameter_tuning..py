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
from tensorflow.keras.layers import Dropout  
from tensorflow.keras.optimizers import Adam  
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("diabetes.csv")
print(df.head())
print(df.isnull().sum())
print(df.duplicated().sum())



sns.heatmap(df.corr(numeric_only=True)[['Outcome']] , annot= True)
plt.show()

X = df.iloc[ :, :-1  ]
y = df.iloc[:, -1]

print( X.head() , y.head())

ss = StandardScaler()

X_scaled= ss.fit_transform(X)
print(X_scaled)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=4)


# model = Sequential()
# model.add( Dense(32 , activation='relu' , input_dim= 8  ))
# model.add(Dropout(0.2))
# model.add( Dense(32 , activation='relu' ))
# model.add(Dropout(0.2))
# model.add( Dense(32 , activation='relu' ))
# model.add(Dropout(0.2))
# model.add( Dense(1 , activation='sigmoid'  ))

# model.compile(loss='binary_crossentropy' , metrics=['accuracy'] , optimizer='Adam')

# history = model.fit(X_train , y_train , validation_data=(X_test , y_test) , epochs = 200 , callbacks = callback)

# y_pred_prob = model.predict(X_test)
# y_pred = np.where(y_pred_prob > 0.5, 1, 0)


# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# plt.plot(history.history['accuracy'] , label='train')
# plt.plot(history.history['val_accuracy'] , label='test')
# plt.legend()
# plt.show()
#                                        1.  hyper parameter tuning    ( best optimizer ko select kese kare )

# def build_model(hp):

#     model = Sequential()
#     model.add(Dense(32 , activation='relu' , input_dim= 8 ))
#     model.add(Dense(1 , activation='sigmoid' ))
#     optimizer = hp.Choice('optimizer' ,values =  ['adam'  , 'sgd' , 'rmsprop' ,'adadelta'])
#     model.compile(optimizer=optimizer , loss='binary_crossentropy' , metrics=['accuracy'])
#     return model


# tuner = kt.RandomSearch(build_model , objective='val_accuracy' , max_trials= 5 , directory='my_tuner', project_name='hp_demo' )

# tuner.search(X_train , y_train , epochs= 10 , validation_data = ( X_test , y_test))
# print(tuner.get_best_hyperparameters()[0].values)   # output : {'optimizer': 'rmsprop'}

# model = tuner.get_best_models(num_models=1)[0]

# print(model.summary())
# callback = EarlyStopping(
#     monitor="val_loss",
#     min_delta=0,
#     patience=0,
#     verbose=0,
#     mode="auto",
#     baseline=None,
#     restore_best_weights=False
# )
# history = model.fit( X_train , y_train ,batch_size= 32 , epochs=100 , initial_epoch= 11 , validation_data =(X_test , y_test)  , callbacks = callback)

# y_pred_prob = model.predict(X_test)
# y_pred = np.where(y_pred_prob > 0.5, 1, 0)


# accuracy = accuracy_score(y_test, y_pred)
# print(accuracy)
#                                            hyperparameter tuning ( optimizer and number  of node kese select kare  )

# def build_model(hp) :
#     model = Sequential()
#     units = hp.Int( 'units', min_value = 8 ,max_value =  128  )
#     model.add( Dense(units = units  , activation = 'relu' , input_dim = 8 ))
#     model.add( Dense(1  , activation = 'sigmoid' ))
#     optimizer = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
#     model.compile(optimizer =optimizer , loss='binary_crossentropy' , metrics=['accuracy'] )
#     return model 

# tuner = kt.RandomSearch(build_model  , objective='val_accuracy' ,max_trials=5 , directory='my_tuner', project_name='hp_demo' )

# tuner.search(X_train , y_train , epochs= 5 , validation_data = ( X_test , y_test))
# print(tuner.get_best_hyperparameters()[0].values)    # it is use to check which parameter is best 
# model = tuner.get_best_models(num_models=1)[0]

# callback = EarlyStopping(
#     monitor="val_loss",
#     min_delta=0,
#     patience=0,
#     verbose=0,
#     mode="auto",
#     baseline=None,
#     restore_best_weights=False
# )
# history = model.fit( X_train , y_train ,batch_size= 32 , epochs=100 , initial_epoch= 11 , validation_data =(X_test , y_test)  )

# y_pred_prob = model.predict(X_test)                
# y_pred = np.where(y_pred_prob > 0.5, 1, 0)


# accuracy = accuracy_score(y_test, y_pred)
# print(accuracy)




#                                                hyper parameter tuning ( number of layer kese select kare )

# def build_model(hp):
#     model = Sequential()

#     model.add(Dense(72 , activation='relu' , input_dim = 8 ))
    
#     for i in range(hp.Int('num_layers' , min_value = 1 , max_value = 10)):
#         model.add(Dropout(0.2))
#         model.add( Dense(72 , activation = 'relu'))

#     model.add(Dense(1 , activation='sigmoid'))

#     # optimizer=hp.Choice('optimizer' , values=['adam' , 'sgd' , 'rmsprop'])
#     model.compile(optimizer = 'rmsprop' , loss='binary_crossentropy' , metrics = ['accuracy'])

#     return model 




# tuner = kt.RandomSearch(build_model , objective='val_accuracy' , max_trials= 5 , directory='my_tuner', project_name='hp_demo' )

# tuner.search(X_train , y_train , epochs= 10 , validation_data = ( X_test , y_test))
# print(tuner.get_best_hyperparameters()[0].values)  

# model = tuner.get_best_models(num_models=1)[0]


# history = model.fit( X_train , y_train ,batch_size= 32 , epochs=100 , initial_epoch= 11 , validation_data =(X_test , y_test)  )



# tuner = kt.RandomSearch(build_model  ,objective='val_accuracy' , max_trials = 5 )

# tuner.search(X_train , y_train ,epochs= 5 , validation_data=(X_test , y_test) )
# print(tuner.get_best_hyperparameters()[0].values) 
# model = tuner.get_best_models(num_models= 1 )[0]

# history = model.fit(X_train ,y_train , validation_data=(X_test , y_test) , epochs= 100 , batch_size = 32  , initial_epoch = 11 )  


#                                                        hyper parameter tuning ( sab kuch )


def build_model(hp):
    model = Sequential()

    counter = 0 
    
    for i in range(hp.Int('num_layers', min_value=1  , max_value=10)):
        
        if counter == 0 :
            model.add(Dense
                (
                    units=hp.Int('units'+str(i), min_value = 8 , max_value= 128 ,step = 8 ) ,
                    activation=hp.Choice('activation' + str(i) , values = ['relu' , 'tanh' , 'sigmoid' ]),
                    input_dim = 8
                ))
            model.add(Dropout(hp.Choice('dropout'+str(i), values=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])))
        else:
            model.add(Dense
                (
                    units=hp.Int('units'+str(i), min_value = 8 , max_value= 128 ,step = 8 ) ,
                    activation=hp.Choice('activation' + str(i) , values = ['relu' , 'tanh' , 'sigmoid' ])
                ))
            model.add(Dropout(hp.Choice('dropout'+str(i), values=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])))
        counter+=1
    model.add(Dense(1 , activation='sigmoid'))
    
    model.compile(optimizer= hp.Choice('optimizer' , values=['rmsprop' , 'adam' , 'sgd' ,'nadam' ,'adadelta']) ,
                    loss='binary_crossentropy' , metrics=['accuracy'])
    
    return model



tuner = kt.RandomSearch(build_model , objective='val_accuracy' , max_trials= 5 , directory='my_tuner', project_name='final' )

tuner.search(X_train , y_train , epochs= 10 , validation_data = ( X_test , y_test))
print(tuner.get_best_hyperparameters()[0].values)  

model = tuner.get_best_models(num_models=1)[0]


history = model.fit( X_train , y_train ,batch_size= 32 , epochs=100 , initial_epoch= 11 , validation_data =(X_test , y_test)  )




    


    
           