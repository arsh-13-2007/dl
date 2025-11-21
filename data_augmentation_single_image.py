import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator  , img_to_array ,array_to_img , load_img


#                        data augmentation 
img = image.load_img(
    r'C:\Users\Arsh\OneDrive\Desktop\dl\single_cat_image\cat_image.jpeg',
    target_size=(200, 200)
)
print(type(img))   # <class 'PIL.Image.Image'>
plt.imshow(img)
plt.axis('off')
plt.show()

datagen = ImageDataGenerator(
    rotation_range= 0.2 , 
    shear_range = 0.2 ,
    zoom_range = 0.2 ,
    horizontal_flip = True, 
    width_shift_range = 0.2 , 
    height_shift_range = 0.2,
    fill_mode='nearest' ,
    # fill_mode='reflect' ,
    # fill_mode='constant'
)


img = image.img_to_array(img)
print(type(img))  #numpy array 
print(img.shape)
input_batch = img.reshape(1,200,200,3)

i= 0 
for output in datagen.flow(input_batch,batch_size = 1 , save_prefix='cat', save_format='jpeg', save_to_dir='aug' ):
    i= i+1 
    if i==20:
        break


print(input_batch.shape)




                               