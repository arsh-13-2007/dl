import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler
import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout , Dense , Flatten , Input , Lambda
from tensorflow.keras.callbacks import EarlyStopping
from  keras.applications.resnet50 import ResNet50 , preprocess_input
from keras.preprocessing.image import ImageDataGenerator , img_to_array , array_to_img , load_img
import kerastuner as kt

batch_size = 15 

train_datagen= ImageDataGenerator(
    rescale= 1./ 255 ,
    shear_range= 0.2 ,
    zoom_range = 0.2,
    horizontal_flip = True 
)

test_datagen = ImageDataGenerator(
    rescale = 1.0 / 255 
)

train_generator= train_datagen.flow_from_directory(
      'car_Datasets\Train' , 
      target_size = (150 , 150),
      batch_size = batch_size,
      class_mode = 'binary'
)

validation=test_datagen.flow_from_directory(
    'car_Datasets\Test'  , 
    target_size = (150 , 150),
    batch_size = batch_size,
    class_mode = 'binary'
)

# ResNet = ResNet50(include_top=False , weights='imagenet' , input_shape=(224, 224, 3))

# print(ResNet.summary())
# from tensorflow.keras.applications import ResNet50

ResNet = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

for layer in ResNet.layers:
   layer.trainable = False
print(ResNet.summary())



