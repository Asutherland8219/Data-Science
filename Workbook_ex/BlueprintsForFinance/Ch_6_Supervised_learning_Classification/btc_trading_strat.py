# this one is important to pay attention too as it pertains to the crypto model i will be building 
# Load libraries
from datetime import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Libraries for Deep Learning Models
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD

#Disable the warnings
import warnings
warnings.filterwarnings('ignore')

''' Load data and peek the dataset '''
dataset = pd.read_csv('Workbook_ex/Datasets/BTCdata.zip', compression='zip', low_memory=True)

print(dataset.head())

## A lot of the data is NaN which is annoying to work with, we should drop it, data super skewed otherwise


dataset = dataset.dropna()

## dropping timestamp because time does not really matter in this instance, we have the data by the minute 

dataset= dataset.drop(columns=['Timestamp'])

print(dataset.head())
print(dataset.shape)
print(dataset.describe())

## Create a simple moving avg of the data 
dataset['short_mavg'] = dataset['Close'].rolling(window=10, min_periods=1, center=False).mean()

## Create a long moving Avg over the long window
dataset['long_mavg'] = dataset['Close'].rolling(window=60, min_periods=1, center=False).mean()

## Create signals 
dataset['signal'] = np.where(dataset['short_mavg'] > dataset['long_mavg'], 1.0, 0.0)

''' Feature Engineering '''
## We will be creating the following momentum indicators:
#  - Moving Avg 
#  - Stochastic Oscillator %K: compares closing price to price over time
#  - Relative Strength Index (RSI): momentum indicator that measures the magnitude of recent price changes to evaluate overbought and oversold
#  - Rate of Change: Change of current price and the n period past prices. High ROC = Overbought, Low ROC = Underbought
#  - Momentum (MOM): Speed at which price is changing 

# calculate exponential moving avg
def EMA(df, n):
    EMA = pd.Series(df['Close'].ewm(span=n, min_periods=n).mean(), name='EMA' + str(n))
    return EMA
dataset['EMA10'] = EMA(dataset, 10)
dataset['EMA30'] = EMA(dataset, 30)
dataset['EMA200'] = EMA(dataset, 200)
print(dataset.head())

# Calculate rate of change 
def ROC(df, n):
    M = df.diff(n-1)
    N = df.shift(n-1)
    ROC = pd.Series(((M / N) * 100), name= 'ROC_' + str(n))
    return ROC
dataset['ROC10'] = ROC(dataset['Close'], 10)
dataset['ROC30'] = ROC(dataset['Close'], 30)

# Calculation of price momentum
def MOM(df, n):
    MOM = pd.Series(df.diff(n), name='Momentum_' + str(n))
    return MOM
dataset['MOM10'] = MOM(dataset['Close'], 10)
dataset['MOM30'] = MOM(dataset['Close'], 30)

# check stop 
print(dataset.head())






