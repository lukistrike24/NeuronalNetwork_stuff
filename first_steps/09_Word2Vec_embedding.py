# -*- coding: utf-8 -*-
"""
Created on Sun May 17 22:11:03 2020

@author: luhoe
"""
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import os

" more info at : https://radimrehurek.com/gensim/models/word2vec.html"

#data loading and preprocessing
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()
 
sentences = MySentences('C:\\Users\\luhoe\\Documents\\Git_Projects\\Github\\NeuronalNetwork_stuff\\first_steps\\training_data\\texts') # a memory-friendly iterator



path = get_tmpfile("C:\\Users\\luhoe\\Documents\\Git_Projects\\Github\\NeuronalNetwork_stuff\\first_steps\\word2vec_models\\word2vec.model")

print('start_training')
model = Word2Vec(sentences, size=200, window=8, min_count=4, workers=6)
print('finished')

model.save("C:\\Users\\luhoe\\Documents\\Git_Projects\\Github\\NeuronalNetwork_stuff\\first_steps\\word2vec_models\\word2vec.model")

