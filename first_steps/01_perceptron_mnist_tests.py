from tensorflow import keras
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import TensorBoard
from time import time
import datetime
import json
import os
import matplotlib.pyplot as plt
from trainingVisualizer import TrainingPlot

    
plot_losses = TrainingPlot()


epochs = 15
optimizer = "adam"
activation_ = "softmax"
hidden_nodes = 10000
loss_ = "categorical_crossentropy"

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

img_width = X_train.shape[1]
img_height = X_train.shape[2]

X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

# one hot encode outputs
y_train = to_categorical(y_train, num_classes=None)
y_test = to_categorical(y_test, num_classes=None)
labels = range(10)

num_classes = y_train.shape[1]

# create model
model=Sequential()
model.add(Flatten(input_shape=(img_width,img_height)))
model.add(Dropout(0.4))
model.add(Dense(hidden_nodes, activation = "relu"))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation = activation_))
model.compile(loss= loss_, optimizer=optimizer,
                    metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), 
      epochs=epochs, callbacks=[plot_losses])

