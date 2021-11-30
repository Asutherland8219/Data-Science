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

# PAD = Padding
# START = Starting point 
# UNK = Unknown point 

word_index["PAD"] = 0
word_index["START"] = 1 
word_index["UNK"] = 2
index_word = {v:k for k,v in word_index.items()}

index_word = ' '.join(index_word[id] for id in x_train[0])

print(index_word)

(all_x_train,_), (all_x_valid,_) = imdb.load_data()


## Standardize the length of reviews
x_train = pad_sequences(x_train, maxlen=max_review_length, padding=pad_type, truncating=trunc_type, value=0)

x_valid = pad_sequences(x_valid, maxlen=max_review_length, padding=pad_type, truncating=trunc_type, value=0)

print(x_train[0:6])

''' Create the dense network '''
model = Sequential()
model.add(Embedding(n_unique_words, n_dim, input_length=max_review_length))
model.add(Flatten())
model.add(Dense(n_dense, activation='relu'))
model.add(Dropout(dropout))

model.add(Dense(1, activation='sigmoid'))

print(model.summary())


# creating the sentiment classifier
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# create the checkpoint to circle back to after each training epoch 
modelcheckpoint = ModelCheckpoint(filepath=output_dir+"/weights.{epoch:02d}.hdf5")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# fit the sentiment classifiier 
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
          validation_data=(x_valid, y_valid), callbacks=[modelcheckpoint])

model.load_weights(output_dir+"/weights.02.hdf5")

y_hat = model.predict(x_valid)

pyplot.hist(y_hat)
_ = pyplot.axvline(x=0.5, color='orange')
pyplot.show()

# calculate ROC AUC for the validation data 
pct_auc = roc_auc_score(y_valid, y_hat)*100
"{:0.2f}".format(pct_auc)

# create a ydf dataframe of y and y hat values 
float_y_hat = []
for y in y_hat:
    float_y_hat.append(y[0])
ydf = pd.DataFrame(list(zip(float_y_hat, y_valid)), columns=['y_hat', 'y'])

# prevew high yhat 
print(ydf[(ydf.y == 0 ) & (ydf.y_hat > 0.9)].head(10))

