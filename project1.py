import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler
import tensorflow 
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout , Dense , Flatten , Input , Lambda
from tensorflow.keras.callbacks import EarlyStopping
from  keras.applications.resnet50 import ResNet50 , preprocess_input
from  keras.applications.vgg16 import VGG16 , preprocess_input
from keras.preprocessing.image import ImageDataGenerator , img_to_array , array_to_img , load_img
import kerastuner as kt

batch_size = 32
image_size = (224, 224)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    shear_range=0.2,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest'
)


test_datagen = ImageDataGenerator(
    rescale = 1.0 / 255 
)

train_generator= train_datagen.flow_from_directory(
      'car_Datasets\Train' , 
      target_size = image_size,
      batch_size = batch_size,
      class_mode='categorical'
)

validation=test_datagen.flow_from_directory(
    'car_Datasets\Test'  , 
    target_size = image_size,
    batch_size = batch_size,
    class_mode='categorical'
)


ResNet = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

for layer in ResNet.layers[-20:]:
    layer.trainable = True


def build_model(hp):
    
    model = Sequential()
    model.add(ResNet)
    model.add(GlobalAveragePooling2D())

    for i in range(hp.Int('num_layers' , min_value=3 , max_value=6)):
        model.add(Dense(
            units = hp.Int('units'+str(i) , min_value= 64 , max_value = 128),
            activation = hp.Choice('activation' + str(i) , values=['relu' , 'sigmoid' , 'tanh' ,'elu'])
            ))
        model.add(Dropout(hp.Choice('dropout_values' + str(i) , values=[0.2,0.3,0.4,0.5,0.6,0.7,0.8])))

        
    model.add(Dense(3 , activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                metrics=['accuracy'] ,
                optimizer= hp.Choice('optimizer' , values=['adam' ,'rmsprop'  ,'adamax' ,'adadelta']
                ))
    

    return model
   
tuner = kt.RandomSearch(build_model , objective='val_accuracy' , max_trials=7 , directory = 'my_tunar' ,project_name='final_result')

tuner.search(train_generator ,epochs = 5 , validation_data=validation,)
print(tuner.get_best_hyperparameters()[0].values)  

model = tuner.get_best_models(num_models=1)[0]
callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=10,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False
)
history = model.fit(train_generator,validation_data=validation,epochs=100,batch_size= 32 , initial_epoch= 10 ,steps_per_epoch=len(train_generator), validation_steps=len(validation) , callbacks = callback)
