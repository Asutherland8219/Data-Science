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
from tensorflow.python.keras.layers.core import SpatialDropout1D




# output directory name 
output_dir = './Deeplearning_illustrated/model_output/rnn'

# training 
epochs = 16
batch_size = 128

# vse 
n_dim = 64
n_unique_words = 10000
max_review_length = 100
pad_type = trunc_type = 'pre'
drop_embed = 0.2

# RNN layer architecture 
n_rnn = 256
drop_rnn = 0.2

(x_train, y_train), (x_valid, y_valid) = imdb.load_data(num_words=n_unique_words)

from keras.layers import SimpleRNN

model = Sequential()
model.add(Embedding(n_unique_words, n_dim, input_length=max_review_length))
model.add(SpatialDropout1D(drop_embed))
model.add(SimpleRNN(n_rnn, dropout=drop_rnn))
model.add(Dense(1, activation='sigmoid'))


