import os
#import cv2
import numpy as np
from PIL import Image

EXT = ".png"
WIDTH = 50
SIZE = (WIDTH, WIDTH)
NUM_IMAGES = 13780
AMOUNT_TRAIN = .8
xtrain = []
xtest = []
ytrain = []
ytest = []

def get_im_array(f, size):
    im = Image.open(f).resize(size)
    return np.array(im)

par_dir = 'cell_images/Parasitized/'
uninf_dir = 'cell_images/Uninfected/'

par_im_files = os.listdir(par_dir)
un_im_files = os.listdir(uninf_dir)


i = 1

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
Y_train = np.array(ytrain)
X_test = np.array(xtest)
Y_test = np.array(ytest)

x_train = X_train.reshape(X_train.shape[0], 50, 50, 3)
#y_train = Y_train.reshape(Y_train.shape[0], 50, 50, 3)
x_test = X_test.reshape(X_test.shape[0], 50, 50, 3)
#y_test = Y_test.reshape(Y_test.shape[0], 50, 50, 3)

print(x_train)
