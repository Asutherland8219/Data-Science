from operator import index
import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
import os 
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd 
import matplotlib.pyplot as plt

# set the hyper params
output_dir = './Deeplearning_illustrated'

# training 
epochs = 4 
batch_size = 128

# vector space embedding 
n_dim = 64
n_unique_words = 5000
n_words_to_skip = 50
max_review_length = 100
pad_type = trunc_type = 'pre'

# neural net architecture 
n_dense = 64
dropout = 0.5

(x_train, y_train), (x_valid, y_valid) = imdb.load_data(num_words=n_unique_words, skip_top=n_words_to_skip)

# print the numbers 
for x in x_train[0:6]:
    print(len(x))
    
word_index = keras.datasets.imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["PAD"] = 0
word_index["START"] = 1 
word_index["UNK"] = 2
index_word = {v:k for k,v in word_index.items()}

index_word = ' '.join(index_word[id] for id in x_train[0])

print(index_word)



