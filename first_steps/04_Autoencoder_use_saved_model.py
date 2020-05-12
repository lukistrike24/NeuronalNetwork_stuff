# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:46:46 2020

@author: luhoe
"""
from tensorflow import keras
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Reshape
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from time import time
import datetime
import json
import os
import matplotlib.pyplot as plt
from trainingVisualizer import TrainingPlot, Images
import cv2


(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

model = load_model('saved_models\\autoencoder_tests.h5')

def add_noise(x_train, noise_factor): 
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0,scale=1.0,size=x_train.shape)
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    return x_train_noisy

fig1 , ax = plt.subplots(1,2) 


i = 0
while(True):
    
    input_img = x_test[i]
        
    output_img = model.predict(input_img.reshape(1,28,28))[0]
    #output_img = output_img[:,:,0]
    ax[0].cla()
    ax[1].cla()
    ax[0].imshow(input_img)
    ax[0].set_title("Input Img")
    ax[1].imshow(output_img)
    ax[1].set_title("Output Img")
    plt.pause(0.05)
    input("Press Enter to continue...")
    i+=1



