from operator import index
import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding
import os 
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd 
import matplotlib.pyplot as pyplot


''' Create and add the CNN '''
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import SpatialDropout1D

# output dir 
output_dir = '/Deeplearning_illustrated/model_output.conv'

# training 
epochs = 4 
batch_size = 128

# vector space embedding 
n_dim = 64
n_unique_words = 5000
n_words_to_skip = 50
max_review_length = 400
pad_type = trunc_type = 'pre'
drop_embed = 0.2 

# conv layer architecture 
n_conv = 256 #filters on the data set, or known as kernels
k_conv = 3  # the length of the kernels 

# dense layer architecture 
n_dense = 256
dropout = 0.2

(x_train, y_train), (x_valid, y_valid) = imdb.load_data(num_words=n_unique_words, skip_top=n_words_to_skip)

model = Sequential()

# vse 
model.add(Embedding(n_unique_words, n_dim, input_length=max_review_length))
model.add(SpatialDropout1D(drop_embed))

# conv layer 
model.add(Conv1D(n_conv, k_conv, activation='relu'))

model.add(GlobalMaxPooling1D())

# dense layer 
model.add(Dense(n_dense, activation='relu'))
model.add(Dropout(dropout))

model.add(Dense(1, activation='sigmoid'))

print(model.summary())

