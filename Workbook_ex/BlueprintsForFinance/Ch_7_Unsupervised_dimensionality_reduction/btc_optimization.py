# this one is important to pay attention too as it pertains to the crypto model i will be building 
# Load libraries
from datetime import time
import numpy as np
import pandas as pd
from matplotlib import pyplot
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
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
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import validation
from sklearn.manifold import TSNE 


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

# standardize the data
scaler = StandardScaler().fit(X_train)
rescaled_Dataset = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns, index = X_train.index)

X_train.dropna(how='any', inplace=True)
rescaled_Dataset.dropna(how='any', inplace=True)
print(rescaled_Dataset.head())

# SVD (Decomposition)
# 5 comps was 94%, 6 was 96$. We are trying 6

ncomps = 6
svd = TruncatedSVD(n_components=ncomps)
svd_fit = svd.fit(rescaled_Dataset)
plt_data = pd.DataFrame(svd_fit.explained_variance_ratio_.cumsum()*100)
plt_data.index = np.arange(1, len(plt_data) + 1)
Y_pred= svd.fit_transform(rescaled_Dataset)
ax = plt_data.plot(kind='line', figsize=(10,4))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel('Eigen Values ')
ax.set_ylabel('Percentage Explained')
ax.legend("")
pyplot.show()
print('Variance preserved by first 6 components == {:.2%}'.format(svd_fit.explained_variance_ratio_.cumsum()[-1]))


dfsvd = pd.DataFrame(Y_pred, columns=['c{}'.format(c) for c in range(ncomps)], index= rescaled_Dataset.index)
print(dfsvd.shape)
print(dfsvd.head())

# visualize the reduced features 
svdcols = [c for c in dfsvd.columns if c[0] == 'c']

plotdims = 5
ploteorows = 1 
dfsvdplot = dfsvd[svdcols].iloc[:, :plotdims]
dfsvdplot['signal']= Y_train
ax = sns.pairplot(dfsvdplot.iloc[::ploteorows, :], hue='signal', size=1.8)
pyplot.show()

# 3d scatter 
def scatter_3D(A, elevation=30, azimuth=120):
    maxpts= 1000
    fig = pyplot.figure(1, figsize=(9,9))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev = elevation, azim= azimuth)
    ax.set_xlabel('component 0')
    ax.set_ylabel('component 1')
    ax.set_zlabel('component 2')

    #plot subset of points 
    rndpts = np.sort(np.random.choice(A.shape[0], min(maxpts, A.shape[0]), replace=False))
    coloridx = np.unique(A.iloc[rndpts]['signal'], return_inverse=True)
    colors = coloridx[1] / len(coloridx[0])
    sp = ax.scatter(A.iloc[rndpts, 0], A.iloc[rndpts, 1], A.iloc[rndpts,2], c= colors, cmap="jet", marker='o', alpha= 0.6, s= 50, linewidths= 0.8, edgecolor = '#BBBBBB')

    pyplot.show()

'''t-SNE visualization '''
tsne = TSNE(n_components = 2 , random_state= 0)
Z = tsne.fit_transform(dfsvd[svdcols])
dftsne = pd.DataFrame(Z, columns=['x', 'y'], index= dfsvd.index)

dftsne['signal'] = Y_train

g = sns.lmplot('x', 'y', dftsne, hue='signal', fit_reg= False, size= 8, scatter_kws= {'alpha':0.7, 's':60})
g.axes.flat[0].set_title('Scatterplot of a Multiple Dimension Dataset reduced to 2D using t-SNE')
pyplot.show()











# ''' Model tuning and grid search '''

# # random forest was the best so we will be using that
# n_estimators = [20,80]
# max_depth = [5,10]
# criterion = ['gini', 'entropy']
# param_grid = dict(n_estimators= n_estimators, max_depth= max_depth, criterion= criterion)
# model= RandomForestClassifier(n_jobs=-1)
# kfold= KFold(n_splits=num_folds, random_state=seed)
# grid= GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
# grid_result = grid.fit(X_train, Y_train)
# print("Best %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# ### optimal model accorging to running Best 0.915762 using {'criterion': 'gini', 'max_depth': 10, 'n_estimators': 80}
# ''' Finalize the model '''
# model = RandomForestClassifier(criterion='gini', n_estimators=80, max_depth=10, n_jobs=-1)

# model.fit(X_train, Y_train)

# ## Estimate the accuracy on the validation set 
# predictions = model.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))

# ## create heatmap of predictions 
# df_cm = pd.DataFrame(confusion_matrix(Y_validation,predictions), columns=np.unique(Y_validation), index= np.unique(Y_validation))
# df_cm.index.name = 'Actual'
# df_cm.columns.name = 'Predicted'
# sns.heatmap(df_cm, cmap='Blues', annot=True, annot_kws={"size":16})
# pyplot.show()

# ## Figure out the variable importance ( how important each feature is to the model )
# Importance = pd.DataFrame({'Importance':model.feature_importances_*100}, index= X.columns)
# Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r')
# pyplot.xlabel('Variable Importance')
# pyplot.show()


# ''' Backtest the data ''' 
# backetestdata = pd.DataFrame(index=X_validation.index)

# backetestdata['signal_pred'] = predictions
# backetestdata['signal_actual'] = Y_validation
# backetestdata['Market Returns'] = X_validation['Close'].pct_change()
# backetestdata['Actual Returns'] = backetestdata['Market Returns'] * backetestdata['signal_actual'].shift(1)
# backetestdata['Strategy Returns'] = backetestdata['Market Returns'] * backetestdata['signal_pred'].shift(1)
# backetestdata = backetestdata.reset_index()
# print(backetestdata.head())

# pyplot.hist(backetestdata[['Strategy Returns', 'Actual Returns']].cumsum())
# pyplot.plot(backetestdata[['Strategy Returns', 'Actual Returns']].cumsum())

# pyplot.show()

