# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 07:40:58 2017

@author: itskh
"""

from keras.models import load_model
from keras.optimizers import Adam
import numpy as np
from scipy.misc import imresize,imread
from PIL import Image

#from keras.preprocessing.image import img_to_array,load_img
model=load_model('best_model_improved.h5')
model.compile(loss='categorical_crossentropy', # Better loss function for neural networks
              optimizer=Adam(lr=1.0e-4),# Adam optimizer with 1.0e-4 learning rate
              metrics = ['accuracy']) #
   
x=imread('2.jpg',mode='RGB')
x=imresize(x,(32,32))
x=np.invert(x)
x=x.reshape(-1,32,32,3)

pred=model.predict(x)
print((pred))