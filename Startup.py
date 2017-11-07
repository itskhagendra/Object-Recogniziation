# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:42:08 2017

@author: itskh
"""

#!/usr/bin/python

print
print ("checking for nltk")
try:
    import tensorflow
except ImportError:
    print ("you should install tensorflow before continuing")

print ("checking for numpy")
try:
    import numpy
except ImportError:
    print ("you should install numpy before continuing")

print ("checking for scipy")
try:
    import scipy
except:
    print ("you should install scipy before continuing")

print ("checking for sklearn")
try:
    import keras
except:
    print ("you should install keras before continuing")
try:
    import matplotlib
except:
    print ("you should install matplotlib before continuing")
try:
    import pickle
except:
    print ("you should install pickle before continuing")    


print ("you're ready to go!")
