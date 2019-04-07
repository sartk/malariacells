#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras
from keras.models import load_model
from keras import backend as K
from PIL import Image

model = load_model('malaria_model.h5')

def processImage(image_path, image_name):
    img = Image.open(image_path+image_name)
    img = img.resize((50,50),resample=0)
    numpyAr = np.array(img).reshape(1,50,50,3)
    return numpyAr
    

def getPrediction(numpyArr):
    if(model.predict(x=numpyArr)>.5):
        return True
    else:
        return False

