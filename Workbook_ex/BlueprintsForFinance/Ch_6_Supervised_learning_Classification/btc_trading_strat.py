# this one is important to pay attention too as it pertains to the crypto model i will be building 
# Load libraries
from datetime import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
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

from sklearn.utils import validation
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

## Going to end up losing more data points early on  due to NaN errors


# Calculation of RSI 
def RSI(series, period):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] ) # sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] ) # sum of avg gains
    d = d.drop(d.index[:(period-1)])
    rs = u.ewm(com=period-1, adjust=False).mean() / d.ewm(com=period-1, adjust=False).mean()
    return 100 - 100 / (1 + rs)
dataset['RSI10'] = RSI(dataset['Close'], 10)
dataset['RSI30'] = RSI(dataset['Close'], 30)
dataset['RSI200'] = RSI(dataset['Close'], 200)

# Calculation of Stocchasting Oscillator
def STOK (close, low, high, n):
    STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    return STOK

def STOD (close, low, high, n):
    STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    STOD = STOK.rolling(3).mean()
    return STOD

dataset['%K10'] = STOK(dataset['Close'], dataset['Low'], dataset['High'], 10)
dataset['%D10'] = STOD(dataset['Close'], dataset['Low'], dataset['High'], 10)
dataset['%K30'] = STOK(dataset['Close'], dataset['Low'], dataset['High'], 30)
dataset['%D30'] = STOD(dataset['Close'], dataset['Low'], dataset['High'], 30)
dataset['%K200'] = STOK(dataset['Close'], dataset['Low'], dataset['High'], 200)
dataset['%D200'] = STOD(dataset['Close'], dataset['Low'], dataset['High'], 200)

# Calculation of moving avg 
def MA(df, n):
    MA = pd.Series(df['Close'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))
    return MA

dataset['MA21'] = MA(dataset, 10)
dataset['MA63'] = MA(dataset, 30)
dataset['MA252'] = MA(dataset, 200)

# stop check

print(dataset.tail())

''' Data Visualization '''

dataset[['Weighted_Price']].plot(grid=True)
pyplot.show()

fig = pyplot.figure()
plot = dataset.groupby(['signal']).size().plot(kind='barh', color='red')
pyplot.show()


''' Evaluate Algorithms and models '''

# split the validation set 
subset_dataset = dataset.iloc[-100000:]
Y = subset_dataset['signal']
X = subset_dataset.loc[:, dataset.columns != 'signal']
validation_size = 0.2
seed = 1
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=1)

num_folds = 10
scoring = 'accuracy'

# # compare models and algos
# models = []
# models.append(('LR', LogisticRegression(n_jobs=-1)))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# # Neural Net 
# models.append(('NN', MLPClassifier()))
# # Ensemble and boosting models 
# models.append(('AB', AdaBoostClassifier()))
# models.append(('GBM', GradientBoostingClassifier()))
# # Bagging method
# models.append(('RF', RandomForestClassifier(n_jobs=-1)))

# ## K-folds cross validation
# results = []
# names = []
# for name, model in models:
#     kfold = KFold(n_splits=num_folds, random_state=seed)
#     cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
#     print(msg)

# ## Visualize the algo comparision 
# fig = pyplot.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# pyplot.boxplot(results)
# ax.set_xticklabels(names)
# fig.set_size_inches(15,8)
# pyplot.show()

''' Model tuning and grid search '''

# random forest was the best so we will be using that
n_estimators = [20,80]
max_depth = [5,10]
criterion = ['gini', 'entropy']
param_grid = dict(n_estimators= n_estimators, max_depth= max_depth, criterion= criterion)
model= RandomForestClassifier(n_jobs=-1)
kfold= KFold(n_splits=num_folds, random_state=seed)
grid= GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X_train, Y_train)
print("Best %f using %s" % (grid_result.best_score_, grid_result.best_params_))

### optimal model accorging to running Best 0.915762 using {'criterion': 'gini', 'max_depth': 10, 'n_estimators': 80}
''' Finalize the model '''
model = RandomForestClassifier(criterion='gini', n_estimators=80, max_depth=10, n_jobs=-1)

model.fit(X_train, Y_train)

## Estimate the accuracy on the validation set 
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

## create heatmap of predictions 
df_cm = pd.DataFrame(confusion_matrix(Y_validation,predictions), columns=np.unique(Y_validation), index= np.unique(Y_validation))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
sns.heatmap(df_cm, cmap='Blues', annot=True, annot_kws={"size":16})
pyplot.show()

## Figure out the variable importance ( how important each feature is to the model )
Importance = pd.DataFrame({'Importance':model.feature_importances_*100}, index= X.columns)
Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r')
pyplot.xlabel('Variable Importance')
pyplot.show()


''' Backtest the data ''' 
backetestdata = pd.DataFrame(index=X_validation.index)

backetestdata['signal_pred'] = predictions
backetestdata['signal_actual'] = Y_validation
backetestdata['Market Returns'] = X_validation['Close'].pct_change()
backetestdata['Actual Returns'] = backetestdata['Market Returns'] * backetestdata['signal_actual'].shift(1)
backetestdata['Strategy Returns'] = backetestdata['Market Returns'] * backetestdata['signal_pred'].shift(1)
backetestdata = backetestdata.reset_index()
print(backetestdata.head())

pyplot.hist(backetestdata[['Strategy Returns', 'Actual Returns']].cumsum())
pyplot.plot(backetestdata[['Strategy Returns', 'Actual Returns']].cumsum())

pyplot.show()