# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 20:02:03 2017

@author: itskh
"""

from flask import Flask,render_template,request
from scipy.misc import imsave,imread,imresize
import numpy as np
import keras.models
import re
from keras.models import model_from_json
import sys
import os
from load import init

sys.path.append(os.path.abspath('./'))


app = Flask(__name__)
global model,graph
model, graph= init()

def convertImage(Img_data1):
    imgstr=re.search(r'base64,(.*)',Img_data1).group(1)
    with open('out.png','wb') as output:
        output.write(imgstr.decode('base64'))
    
    
@app.route('/')
def index():
    return 'Hello World'
@app.route('/predict',methods=['GET','POST'])
def predict():
    Img_Data=request.get_data()
    convertImage(Img_Data)
    x=imread('out.png',mode='L')
    x=imresize(x,32,32)
    x=x.reshape(-1,3,32,32)
    x=x.transpose([0, 2, 3, 1])
    with graph.as_default():
        out=model.predict(x)
        response=np.array_str(np.argmax(out,axis=1))
        return response
    
app.run(debug=True,port=8080)