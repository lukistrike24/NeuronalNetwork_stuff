# -*- coding: utf-8 -*-
"""
Created on Wed May 13 12:32:12 2020

@author: luhoe
"""
from tensorflow import keras
import numpy as np
from tensorflow.keras.datasets import mnist, imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D, Conv2D, GlobalMaxPooling1D, MaxPooling2D, Embedding, LSTM, Bidirectional, Reshape
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing import text, sequence
from time import time
from trainingVisualizer import TrainingPlot
import os


#parameters
vocab_size = 5000
maxlen = 300
batch_size = 32
embedding_dims = 50
filters = 100
kernel_size = 2
hidden_dims = 10
epochs = 8


#only used for dataloadin
def load_imdb():
    X_train = []
    y_train = []
    
    print('loading set 1 of 4 ...')
    path = 'training_data\\aclImdb\\train\\pos\\'
    X_train.extend([open(path + f,  encoding="utf8").read() for f in os.listdir(path) if f.endswith('.txt')])
    y_train.extend([1 for _ in range(12500)])
    
    print('loading set 2 of 4 ...')
    path = 'training_data\\aclImdb\\train\\neg\\'
    X_train.extend([open(path + f,  encoding="utf8").read() for f in os.listdir(path) if f.endswith('.txt')])
    y_train.extend([0 for _ in range(12500)])

    X_test = []
    y_test = []
    
    print('loading set 3 of 4 ...')
    path = 'training_data\\aclImdb\\test\\pos\\'
    X_test.extend([open(path + f,  encoding="utf8").read() for f in os.listdir(path) if f.endswith('.txt')])
    y_test.extend([1 for _ in range(12500)])
    
    print('loading set 4 of 4 ...')
    path = 'training_data\\aclImdb\\test\\neg\\'
    X_test.extend([open(path + f,  encoding="utf8").read() for f in os.listdir(path) if f.endswith('.txt')])
    y_test.extend([0 for _ in range(12500)])
    
    print('done loading')
    return (X_train, y_train), (X_test, y_test)

# load data
(X_train, y_train), (X_test, y_test) = load_imdb()

tokenizer = text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_matrix(X_train)
X_test = tokenizer.texts_to_matrix(X_test)

X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(vocab_size,
                    embedding_dims,
                    input_length=maxlen))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Reshape((250,1)))
model.add(Dropout(0.2))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.2))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

X_test = np.array(X_test)
y_test = np.array(y_test)
X_train = np.array(X_train)
y_train = np.array(y_train)

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test), callbacks=[TrainingPlot()])