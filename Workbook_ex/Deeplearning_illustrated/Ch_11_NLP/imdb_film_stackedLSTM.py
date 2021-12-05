from operator import index
import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Embedding
from keras.layers.wrappers import Bidirectional
import os 
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd 
import matplotlib.pyplot as pyplot


''' Create and add the CNN '''
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import SpatialDropout1D

# output dir 
output_dir = '/Deeplearning_illustrated/model_outputs'

# training 
epochs = 4 
batch_size = 128

# vector space embedding 
n_dim = 64
n_unique_words = 10000
max_review_length = 100
pad_type = trunc_type = 'pre'
drop_embed = 0.2 

# LSTM layer architecture 
n_lstm_1 = 256
n_lstm_2 = 256#filters on the data set, or known as kernels
drop_lstm = 0.2   



(x_train, y_train), (x_valid, y_valid) = imdb.load_data(num_words=n_unique_words)

model = Sequential()
model.add(Embedding(n_unique_words, n_dim, input_length=max_review_length))
model.add(SpatialDropout1D(drop_embed))
model.add(Bidirectional(LSTM(n_lstm_1, dropout=drop_lstm, return_sequences=True)))
model.add(Bidirectional(LSTM(n_lstm_2, dropout=drop_lstm)))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())