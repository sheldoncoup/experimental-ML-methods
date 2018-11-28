import numpy as np

# Sys
import warnings
# Keras Core
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D

from keras.layers.merge import concatenate
from keras import regularizers
from keras import initializers
from keras.models import Model
# Backend
from keras import backend as K
# Utils


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2



def preprocess_input(x):
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    x = np.multiply(x, 2.0)
    return x

def build_alexnet(num_classes,img_size, dropout_prob, weights, phase, freeze_layers, l2_reg=0.):
    img_shape = (img_size, img_size,3)
    # Initialize model
    alexnet = Sequential()
    if freeze_layers:
        if phase == 0:
            tr = True
        else:
            tr = False
    else:
        tr = True   
    # Layer 1
    alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape, strides=(4,4),padding='same', kernel_regularizer=l2(l2_reg),trainable=tr))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    
    if phase >= 1 or not freeze_layers:
        if freeze_layers:
            if phase == 1:
                tr = True
            else:
                tr = False
        else:
            tr = True
        # Layer 2
        alexnet.add(Conv2D(256, (5, 5), padding='same',trainable=tr ))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    
    if phase >= 2 or not freeze_layers :
        if freeze_layers:
            if phase == 2:
                tr = True
            else:
                tr = False
        else:
            tr = True 
        # Layer 3
        alexnet.add(ZeroPadding2D((1, 1)))
        alexnet.add(Conv2D(512, (3, 3), padding='same',trainable=tr))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(MaxPooling2D(pool_size=(2, 2)))
    if phase >= 3 or not freeze_layers: 
        if freeze_layers:
            if phase == 3:
                tr = True
            else:
                tr = False
        else:
            tr = True
        # Layer 4
        alexnet.add(ZeroPadding2D((1, 1)))
        alexnet.add(Conv2D(1024, (3, 3), padding='same',trainable=tr))
        alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
    
    if phase >= 4 or not freeze_layers:
        if freeze_layers:
            if phase == 4:
                tr = True
            else:
                tr = False
        else:
            tr = True
            # Layer 5
            alexnet.add(ZeroPadding2D((1, 1)))
            alexnet.add(Conv2D(1024, (3, 3), padding='same',trainable=tr))
            alexnet.add(BatchNormalization())
            alexnet.add(Activation('relu'))
            alexnet.add(MaxPooling2D(pool_size=(2, 2)))
  
    # Layer 6
    alexnet.add(Flatten())
    alexnet.add(Dense(3072))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(dropout_prob))
    
    # Layer 7
    alexnet.add(Dense(4096))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(dropout_prob))
    
    # Layer 8
    alexnet.add(Dense(num_classes))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('softmax'))
    
    if weights == 'random':
        pass
    elif weights is not None:
        alexnet.load_weights(weights,by_name=True, skip_mismatch=True)
  
    return alexnet


def create_model(num_classes=1001,img_size=227, dropout_prob=0.2, weights=None, include_top=True, phase=4, freeze_layers=False):
    return build_alexnet(num_classes,img_size, dropout_prob, weights, phase, freeze_layers)
