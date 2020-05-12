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
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from time import time
import datetime
import json
import os
import matplotlib.pyplot as plt
from trainingVisualizer import TrainingPlot, Images

    
#plot_losses = TrainingPlot()

epochs_ = 1000
encoding_dim = 25


(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(encoding_dim, activation='relu'))
model.add(Dense(28*28, activation='sigmoid'))
model.add(Reshape((28,28)))
model.compile(optimizer='adam', loss='mse')

model.summary()

model.fit(x_train, x_train,
                epochs=epochs_,
                validation_data=(x_test, x_test), batch_size = 1024, 
                callbacks=[TrainingPlot()])
                #callbacks=[Images((x_test, x_test)), TrainingPlot()])


#model.save('saved_models\\autoencoder_tests.h5')


