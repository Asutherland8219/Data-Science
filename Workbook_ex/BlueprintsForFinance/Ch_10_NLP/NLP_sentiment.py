import pkg_resources
import pip

#NLP libraries
from textblob import TextBlob, tokenizers
import spacy
import yfinance as yf

import nltk
import warnings
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
import csv
import pandas as pd

#Run the command python -m spacy download en_core_web_sm to download this
#https://spacy.io/models

nlp = spacy.load('en_core_web_sm')

#Libraries for processing the news headlines
from lxml import etree
import json
from io import StringIO
from os import listdir
from os.path import isfile, join
from pandas.tseries.offsets import BDay
from scipy.stats.mstats import winsorize
from copy import copy

# Libraries for Classification for modeling the sentiments
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Keras package for the deep learning model for the sentiment prediction. 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout, Activation
from keras.layers.embeddings import Embedding

# Load libraries
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
from datetime import date
import matplotlib.pyplot as pyplot


#Additional Libraries 
import json  
import zipfile
import os.path
import sys

import warnings
warnings.filterwarnings('ignore')

# tickers = pd.read_json('Workbook_ex/BlueprintsForFinance/Ch_10_NLP/tickers.json')
tickers = ["AAPL", "MSFT", "FB", "WMT", "JPM", "TSLA", "GOOG", "AMZN", "NFLX", "ADBE"]
start = '2010-01-01'
end = '2018-12-31'

df_ticker_return = pd.DataFrame()

for ticker in tickers:
    ticker_yf = yf.Ticker(ticker)
    if df_ticker_return.empty:
        df_ticker_return = ticker_yf.history(start=start, end=end)
        df_ticker_return['ticker'] = ticker
    else:
        data_temp = ticker_yf.history(start=start, end=end)
        data_temp['ticker']= ticker
        df_ticker_return = df_ticker_return.append(data_temp)
df_ticker_return.to_csv('Workbook_ex/Datasets/NLP_sets/returndata.csv')

print(df_ticker_return.head())

z = zipfile.ZipFile(r"Workbook_ex\Datasets\NLP_sets\Raw Headline Data.zip")
testFile = z.namelist()[10]
fileData = z.open(testFile).read()
fileDataSample = json.loads(fileData)['content'][1:500]
print(fileDataSample)

''' Create a Json parser to parse the data '''
def jsonParser(json_data): 
    xml_data = json_data['content']
            
    tree = etree.parse(StringIO(xml_data), parser=etree.HTMLParser())

    headlines = tree.xpath("//h4[contains(@class, 'media-heading')]/a/text()")
    assert len(headlines) == json_data['count']

    main_tickers = list(map(lambda x: x.replace('/symbol/', ''), tree.xpath("//div[contains(@class, 'media-left')]//a/@href")))
    assert len(main_tickers) == json_data['count']
    final_headlines = [''.join(f.xpath('.//text()')) for f in tree.xpath("//div[contains(@class, 'media-body')]/ul/li[1]")]
    if len(final_headlines) == 0:
        final_headlines = [''.join(f.xpath('.//text()')) for f in tree.xpath("//div[contains(@class, 'media-body')]")]
        final_headlines = [f.replace(h, '').split('\xa0')[0].strip() for f,h in zip (final_headlines, headlines)]
    return main_tickers, final_headlines

data = None 
data_df_news = []
ret = []
ret_f = []
with zipfile.ZipFile("Workbook_ex/Datasets/NLP_sets/Raw Headline Data.zip", "r") as z:
    for filename in z.namelist(): 
        #print(filename)
        try:               
            #print('Running {}'.format(filename))
            with z.open(filename) as f:  
                data = f.read()  
                json_data = json.loads(data)      
            if json_data.get('count', 0)> 10:
                #Step 1: Parse the News Jsons 
                main_tickers, final_headlines = jsonParser(json_data) 
                if len(final_headlines) != json_data['count']:
                    continue
                #Step 2: Prepare Future and Event Return and assign Future and Event return for each ticker. 
                file_date = filename.split('/')[-1].replace('.json', '')
                file_date = date(int(file_date[:4]), int(file_date[5:7]), int(file_date[8:]))
               #Step 3: Merge all the data in a data frame
                df_dict = {'ticker': main_tickers,
                           'headline': final_headlines,            
                           'date': [file_date] * len(main_tickers)
                           }
                df_f = pd.DataFrame(df_dict)            
                data_df_news.append(df_f)            
        except:
            pass  

data_df_news = pd.concat(data_df_news)
print(data_df_news.head())

''' Preparing the combined data '''
df_ticker_return['ret_curr'] = df_ticker_return['Close'].pct_change()

#Event return
df_ticker_return['eventRet'] = df_ticker_return['ret_curr'] + df_ticker_return['ret_curr'].shift(-1) + df_ticker_return['ret_curr'].shift(1)

df_ticker_return.reset_index(level=0, inplace=True)
df_ticker_return['date'] = pd.to_datetime(df_ticker_return['Date']).apply(lambda x: x.date())

combinedDataFrame = pd.merge(data_df_news, df_ticker_return, how='left', left_on=['date', 'ticker'], right_on=['date', 'ticker'])
combinedDataFrame = combinedDataFrame[combinedDataFrame['ticker'].isin(tickers)]
data_df = combinedDataFrame[['ticker', 'headline', 'date', 'eventRet', 'Close']]
data_df = data_df.dropna()

print(data_df.head()
)

''' Save the data for use later '''
data_df.dropna().to_csv(r'Workbook_ex/Datasets/NLP_sets/sentiment.csv', sep='|', index=False)

''' If you want to load pre saved data use this'''
# data_df = pd.read_csv(r'Workbook_ex/Datasets/NLP_sets/sentiment.csv', sep='|')
# data_df = data_df.dropna()
# print(data_df.shape, data_df.ticker.unique().shape)

''' Begin Creating model for analysis '''
# predefined 
text1 = "Bayer (OTCPK:BAYRY) started the week up 3.5% to €74/share in Frankfurt, touching their \
highest level in 14 months, after the U.S. government said a $25M glyphosate decision against the \
company should be reversed."

print(TextBlob(text1).sentiment.polarity)

#graph the data 

data_df['sentiment_textblob'] = [TextBlob(s).sentiment.polarity for s in data_df['headline']]

pyplot.scatter(data_df['sentiment_textblob'], data_df['eventRet'], alpha=0.5)
pyplot.title('Scatter Between Event Return and Sentiments- all data')
pyplot.ylabel('Event Return')
pyplot.xlabel('Sentiments')
pyplot.show()

#correlation graph 
correlation = data_df['eventRet'].corr(data_df['sentiment_textblob'])
print(correlation)

data_df_stock = data_df[data_df['ticker'] == 'AAPL' ]
pyplot.scatter(data_df['sentiment_textblob'], data_df_stock['eventRet'], alpha=0.5)
pyplot.title('Scatter Between Event Return and Sentiments-AAPL')
pyplot.ylabel('Event Return')
pyplot.xlabel('Sentiments')
pyplot.show()


''' Creating our own LSTM model -- Supervised'''
# this done using a premade dataset from kaggle.com
sentiments_data = pd.read_csv(r'Workbook_ex/Datasets/NLP_sets/LabelledNewsData.csv', encoding='ISO-8859-1')
print(sentiments_data.head())

all_vectors = pd.np.array([pd.np.array([token.vector for token in nlp(s) ])]).mean(axis=0)*pd.np.ones((300)) \
    for s in sentiments_data['headline']])

Y = sentiments_data['sentiment']
X = all_vectors

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
validation_size = 0.3
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=validation_size, random_state=seed)

num_folds = 10
seed = 7
scoring = 'accuracy'

# check the algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))

# Neural Networks
models.append(('NN', MLPClassifier()))

# Ensemble models 
models.append(('RF', RandomForestClassifier()))

## Running each model
results = []
names = []
kfold_results = []
test_results = []
train_results = []

for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)

    # training period
    res = model.fit(X_train, y_train)
    train_result = accuracy_score(res.predict(X_train), y_train)
    train_results.append(train_result)

    # test results 
    test_result = accuracy_score(res.predict(X_test), y_test)
    test_results.append(test_result)

    msg = '%s: %f (%f) %f %f' % (name, cv_results.mean(), cv_results.std(), train_result, test_result)
    print(msg)
    print(confusion_matrix(res.predict(X_test), y_test))

# graph the comparison
fig = pyplot.figure()
ind = np.arange(len(names)) 
width = 0.35
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.bar(ind - width/2, train_results, width=width, label='Train Error')
pyplot.bar(ind + width/2, test_results, width=width, label='Test Error')

fig.set_size_inches(15, 8)
pyplot.legend()
ax.set_xticks(ind)
ax.set_xticklabels(names)
pyplot.show()


''' LSTM model '''
### Create the sequence
vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(sentiments_data['headline'])
sequences = tokenizer.texts_to_sequences(sentiments_data['headline'])
X_LSTM = pad_sequences(sequences, maxlen=50)

y_LSTM =  sentiments_data["sentiment"]
X_train_LSTM, X_test_LSTM, y_train_LSTM, y_test_LSTM = train_test_split(X_LSTM, y_LSTM, test_size=validation_size, random_state=seed)

# Use keras to build the neural network classifier
from keras.wrappers.scikit_learn import KerasClassifier
def create_model(input_length=50):
    model = Sequential()
    model.add(Embedding(20000, 300, input_length=50))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model 

model_LSTM = KerasClassifier(build_fn= create_model, epochs=3, verbose=1, validation_split=0.4)
model_LSTM.fit(X_train_LSTM, y_train_LSTM)







