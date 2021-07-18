''' Sentiment analysis using IMDB movie review dataset '''
# We are using the pre labelled keras data set 

from tensorflow.keras.datasets import imdb
from tensorflow.python.keras.backend import random_bernoulli
from tensorflow.python.keras.layers.embeddings import Embedding 
number_of_words = 10000

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=number_of_words
)

''' Decoding each movie review '''
word_to_index = imdb.get_word_index()
print(word_to_index['amazing'])

index_to_word = {index: word for (word, index) in word_to_index.items()}

''' Data Preparation '''
words_per_review = 200 
from tensorflow.keras.preprocessing.sequence import pad_sequences
X_train = pad_sequences(X_train, maxlen=words_per_review)
X_test = pad_sequences(X_test, maxlen=words_per_review)


from sklearn.model_selection import train_test_split
X_test, X_val, y_test, y_val = train_test_split(
    X_test, y_test, random_state=11, test_size=0.20
)


''' Create the RNN '''
from tensorflow.keras.models import Sequential

rnn = Sequential()

from tensorflow.keras.layers import Dense, LSTM

rnn.add(Embedding(input_dim=number_of_words, output_dim=128, input_length=words_per_review))
rnn.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))





