from tensorflow import keras
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from time import time
import datetime
import json
import os
import matplotlib.pyplot as plt
from trainingVisualizer import TrainingPlot

    
plot_losses = TrainingPlot()


epochs = 100
optimizer = "adam"
activation_ = "relu"
hidden_nodes = 750
loss_ = "categorical_crossentropy"
first_layer_conv_width = 5
first_layer_conv_height = 5
dense_layer_size = 100

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#normalization
X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

#get img dimensions
img_width = X_train.shape[1]
img_height = X_train.shape[2]

#reshape input data
X_train = X_train.reshape(X_train.shape[0], img_width, img_height, 1)
X_test = X_test.reshape(X_test.shape[0], img_width, img_height, 1)

# one hot encode outputs
y_train = to_categorical(y_train, num_classes=None)
y_test = to_categorical(y_test, num_classes=None)
labels = range(10)

num_classes = y_train.shape[1]

# create model
model=Sequential()
model.add(Conv2D(32,(first_layer_conv_width, first_layer_conv_height),input_shape=(28, 28, 1),activation = activation_))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(32,(first_layer_conv_width, first_layer_conv_height),input_shape=(28, 28, 1),activation = activation_))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.4))
model.add(Dense(hidden_nodes, activation = "relu"))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(dense_layer_size, activation=activation_))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation = "softmax"))
model.compile(loss= loss_, optimizer=optimizer,
                    metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), 
      epochs=epochs, callbacks=[plot_losses])

