
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
import tensorflow as tf
import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.datasets import fashion_mnist
import  kerastuner as kt 



(X_train , y_train ) , (X_test , y_test ) = fashion_mnist.load_data()


#                        preprocessing the data 
X_train = X_train / 255
X_test = X_test / 255


print(X_train[0].shape)

x_train =X_train.reshape(len(X_train) , 28 , 28 ,1 )
x_test =X_test.reshape(len(X_test) , 28 , 28 ,1 )

# model =  Sequential()

# model.add(Conv2D(32 , padding='valid',  activation='relu' , input_shape = (256 , 256, 3)))
# model.add(MaxPooling2D(padding='valid' , pool_size=(2,2) , strides=2))
# model.add(Conv2D(32 , activation='relu' , padding='valid'))
# model.add(MaxPooling2D(padding='valid' , pool_size=(2,2) , strides=2))
# model.add(Conv2D(32 , activation='relu' , padding='valid'))
# model.add(MaxPooling2D(padding='valid' , pool_size=(2,2) , strides=2))


# model.add(Flatten())

# model.add( Dense(128 , acitvation='relu' ))
# model.add( dense(128  , activation='relu'))
# model.add( Dense(64 , activation='relu '))
# model.add( Dense(1, activation='sigmoid'))

# model.


def build_model(hp):
    model = Sequential()
    counter = 0 

    for i in range(hp.Int('num_layers', min_value=1  , max_value=5)):
        if counter == 0:
            model.add( Conv2D(                hp.Int('num_filter' , min_value=64  , max_value=128  ) , 
                activation=hp.Choice('activation'+ str(i) , values=['relu' , 'sigmoid' , 'tanh']) , 
                kernel_size=(3,3) , 
                padding= hp.Choice('padding' + str(i) , values=['valid' , 'same']) , 
                input_shape=(28 , 28 , 1 )
            ))
            model.add( MaxPooling2D(
                padding= hp.Choice('padding' + str(i) , values=['valid' , 'same']),
                strides = hp.Choice(f"strides_{i}", values=[1, 2]),
                pool_size=(2,2)
            ))

        else: 
            model.add( Conv2D(
                hp.Int('num_filter' , min_value=64  , max_value=128  ) , 
                activation=hp.Choice('activation'+ str(i) , values=['relu' , 'sigmoid' , 'tanh']) , 
                kernel_size=(3,3) , 
                padding= hp.Choice('padding' + str(i) , values=['valid' , 'same']) , 
            ))
            model.add( MaxPooling2D(
                padding= hp.Choice('padding' + str(i) , values=['valid' , 'same']),
                strides = hp.Choice(f"strides_{i}", values=[1, 2]),
                pool_size=(2,2)
            ))
        counter += 1
    model.add(Flatten())     # flatten the means  convert it 2d or 3d into 1d 
    for i in range(hp.Int('num_layers', min_value=1  , max_value=10)):
        model.add(Dense
            (
                units=hp.Int('units'+str(i), min_value = 8 , max_value= 128 ,step = 8 ) ,
                activation=hp.Choice('activation' + str(i) , values = ['relu' , 'tanh' , 'sigmoid' ])
            ))
        model.add(Dropout(hp.Choice('dropout'+str(i), values=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])))
    model.add(Dense(10, activation='softmax'))
      
    model.compile(optimizer= hp.Choice('optimizer' , values=['rmsprop' , 'adam' , 'sgd' ,'nadam' ,'adadelta']) ,
                   loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


tuner = kt.RandomSearch(build_model , objective='val_accuracy' , max_trials= 5 , directory='my_tuner', project_name='final' )

tuner.search(X_train , y_train , epochs= 4 , validation_split = 0.2 )
print(tuner.get_best_hyperparameters()[0].values)  

model = tuner.get_best_models(num_models=1)[0]
callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=2,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False
)
history = model.fit( X_train , y_train ,batch_size= 32 , epochs=12 , initial_epoch= 4 , validation_data =(X_test , y_test), callbacks = callback  )



    

