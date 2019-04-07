#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np #lin. Alg.
import pandas as pd #data processing
import matplotlib.pyplot as plt
import matplotlib.image as matimg

#importing keras modules
import keras
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout
from keras.layers import Activation
from keras.models import load_model
from keras import backend as K
from keras import optimizers

#Python Image Library for image processing
from PIL import Image

import os

#Path of parasite and uninfected folders
image_path="D:malariaData/cell_images/"



# In[78]:


EXT = ".png"
WIDTH = 50 #Width of the transformed image
SIZE = (WIDTH, WIDTH) #Square final image
NUM_IMAGES = 13780 
AMOUNT_TRAIN = .8

#Splitting data into test and train sets
xtrain = []
xtest = []
ytrain = []
ytest = []

#Opens particular image, resizes it, and returns a numpy array of it
def get_im_array(f, size):
    im = Image.open(f).resize(size)
    return np.array(im)

#directories of infected and uninfected images
par_dir = image_path + 'Parasitized/'
uninf_dir = image_path+ 'Uninfected/'

#files in each directory
par_im_files = os.listdir(par_dir)
un_im_files = os.listdir(uninf_dir)



i = 1 #iterator


for f in par_im_files:
    if f.endswith(EXT):
        if i <= NUM_IMAGES*AMOUNT_TRAIN:
            xtrain.append(get_im_array(par_dir+f, SIZE))
            ytrain.append(1)
        else:
            xtest.append(get_im_array(par_dir+f, SIZE))
            ytest.append(1)
    i+=1

i = 1

for f in un_im_files:
    if f.endswith(EXT):
        if i <= NUM_IMAGES*AMOUNT_TRAIN:
            xtrain.append(get_im_array(uninf_dir+f, SIZE))
            ytrain.append(0)
        else:
            xtest.append(get_im_array(uninf_dir+f, SIZE))
            ytest.append(0)
    i+=1


X_train = np.array(xtrain)
y_train = np.array(ytrain)
X_test = np.array(xtest)
y_test = np.array(ytest)

x_train = X_train.reshape(X_train.shape[0], 50, 50, 3)/255.
x_test = X_test.reshape(X_test.shape[0], 50, 50, 3)/255.


# In[81]:


#HyperParamters
#Max Pooling Sizes
m_1 = 2
m_4 = 2
#Dropout values of the layers
d_1 = .3
d_2 = .2
#kernel sizes
k_1 = 3
k_2 = 3
k_3 = 3
k_4 = 3
#Convolution layer sizes
c_1 = 32
c_2 = 64
c_3 = 64
c_4 = 32


# In[83]:


data_format = 'channels_last'

#Building model
model = Sequential()

#Block1
model.add(Conv2D(c_1, kernel_size=(k_1, k_1), input_shape=x_train.shape[1:], activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(m_1,m_1)))

#Block2
model.add(Conv2D(c_2, kernel_size=(k_2, k_2), activation='relu', padding='same'))

#Block3
model.add(Conv2D(c_3, kernel_size=(k_3, k_3), activation='relu', padding='same'))

#Block4
model.add(Conv2D(c_4, kernel_size=(k_4, k_4), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(m_4,m_4)))

#Block5
model.add(Flatten())
model.add(Dropout(n1))
model.add(Dense(128, activation='relu'))

#Block6
model.add(Dropout(n2))
model.add(Dense(1, activation='sigmoid'))

#Compiling the model
model.compile(keras.optimizers.Adam(), loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

keras.utils.print_summary(model)


# In[30]:


#Number of epochs to train with
epoch_num = 25

model_log = model.fit(x_train, y_train, epochs=epoch_num, batch_size=2000, verbose=1, validation_data=(x_test, y_test), shuffle=True)


# In[87]:


#Plot accuracy vs Epochs
plt.plot(model_log.history['acc'])
plt.plot(model_log.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[32]:


#Save a Keras model as an .h5 file
model.save('malaria_model.h5')


# In[85]:


#Load a Keras model from a .h5 file
model = load_model('malaria_model.h5')


# In[90]:


#Shows the image and prediction for the 'view_num' element of the test set.
view_num = 100

print('Prediction: ')
print(model.predict(x_test[view_num:view_num+1]))
print()
print('Correct: ')
print(y_test[view_num])

#Displays the image
img = Image.fromarray((X_test.reshape(X_test.shape[0], 50, 50, 3))[view_num], 'RGB')
im_array = np.asarray(img)
plt.imshow(im_array)
plt.show()


# In[ ]:




