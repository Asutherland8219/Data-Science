''' Function and modules for the Supervised Regression Models '''
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor

''' Function and modules for Data Analysis and Model Evaluation '''
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression

''' Function and modules for deep learning models '''
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasRegressor

'''Function and modules for  time series models'''
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm 

'''Function and modules for data preparation and visualization'''

# standard data science imports
import numpy as np
import pandas as pd 
import pandas_datareader.data as web
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
from statsmodels.graphics.tsaplots import plot_acf

#load the data
stk_tickers = ['MSFT', 'IBM', 'GOOGL']
ccy_tickers = ['DEXJPUS', 'DEXUSUK']
idx_tickers = ['SP500', 'DJIA', 'VIXCLS']

stk_data = web.DataReader(stk_tickers, 'yahoo')
ccy_data = web.DataReader(ccy_tickers, 'fred')
idx_data = web.DataReader(idx_tickers, 'fred')

# need to use 5 day lag as only 5 trading days a week

return_period = 5
Y = np.log(stk_data.loc[:, ('Adj Close', 'MSFT')]).diff(return_period).\
shift(-return_period)
Y.name = Y.name[-1]+'_pred'

X1 = np.log(stk_data.loc[:,('Adj Close', ('GOOGL', 'IBM'))]).diff(return_period)
X1.columns = X1.columns.droplevel()
X2 = np.log(ccy_data).diff(return_period)
X3 = np.log(idx_data).diff(return_period)

X4 = pd.concat([np.log(stk_data.loc[:, ('Adj Close', 'IBM')]).diff(i) \
    for i in [return_period, return_period * 3, \
        return_period*6, return_period*12]], axis=1).dropna()
X4.columns = ['MSFT_DT', 'MSFT_3DT', 'MSFT_6DT', 'MSFT_12DT']

X = pd.concat([X1, X2, X3, X4], axis=1)

dataset = pd.concat([Y, X], axis=1).dropna().iloc[::return_period, :]
Y = dataset.loc[:, Y.name]
X = dataset.loc[:, X.columns]

''' Data visualization '''
# Correlation Matrix chart 
correlation = dataset.corr()
pyplot.figure(figsize=(15,15))
pyplot.title('Correlation Matrix')
sns.heatmap(correlation, vmax = 1, square=True, annot=True, cmap='cubehelix')
pyplot.show()

# Scatter plot 
pyplot.figure(figsize=(15,15))
scatter_matrix(dataset, figsize=(12,12))
pyplot.show()

# Time Series Analysis 
res = sm.tsa.seasonal_decompose(Y, period=52)
fig = res.plot()
fig.set_figheight(8)
fig.set_figwidth(15)
pyplot.show()

''' Create the train, test, split and ML Model ''' 
validation_size = 0.2
train_size = int(len(X) * (1-validation_size))
X_train, X_test = X[0:train_size], X[train_size:len(X)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

#Cross validation to test hyperparameters
num_folds = 10
scoring = 'neg_mean_squared_error'

# Model Selection 
## Regression and tree regression algorithms 
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

## Neural NEtwork Algorithms
models.append(('MLP', MLPRegressor()))

## Ensemble Models 

### Boosting Methods 
models.append(('ABR', AdaBoostRegressor()))
models.append(('GBR',GradientBoostingRegressor()))

### Bagging Methods 
models.append(('RFR', RandomForestRegressor()))
models.append(('ETR', ExtraTreesRegressor()))

''' Tuning and Results of Cross Validation ''' 
names = []
kfold_results = []
test_results = []
train_results = []
for name, model in models :
    names.append(name)
    ## K-Fold Analysis
    kfold = KFold(n_splits=num_folds)
    ## Convert MSE to positive, lower = better
    cv_results = -1*cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    kfold_results.append(cv_results)
    # Full training period 
    res = model.fit(X_train, Y_train)
    train_result = mean_squared_error(res.predict(X_train), Y_train)
    train_results.append(train_result)
    # Test results 
    test_result = mean_squared_error(res.predict(X_test), Y_test)
    test_results.append(test_result)

### Compare CV results 
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison: KFold results')
ax = fig.add_subplot(111)
pyplot.boxplot(kfold_results)
ax.set_xticklabels(names)
fig.set_size_inches(15,8)
pyplot.show()

''' Training and test error '''
# comparing algorithms
fig = pyplot.figure()

ind = np.arange(len(names)) # x axis location 
width = 0.35 #bar width 

fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.bar(ind - width/2, train_results, width=width, label='Train Error')
pyplot.bar(ind + width/2, test_results, width=width, label='Test Error')
fig.set_size_inches(15,8)
pyplot.legend()
ax.set_xticks(ind)
ax.set_xticklabels(names)
pyplot.show()

''' Time Series Based Models with ARIMA and LSTM '''
### lok at this in more depth, needs more work ###
X_train_ARIMA = X_train.loc[:,['GOOGL', 'IBM', 'DEXJPUS', 'SP500', 'DJIA', 'VIXCLS']]
X_test_ARIMA = X_test.loc[:,['GOOGL', 'IBM', 'DEXJPUS', 'SP500', 'DJIA', 'VIXCLS']]
tr_len = len(X_train_ARIMA)
te_len = len(X_test_ARIMA)
to_len = len(X)

# Use the ARIMAX MODEL where X represents the exogenous models 
modelARIMA= ARIMA(endog=Y_train, exog=X_train_ARIMA, order=[1,0,0])
model_fit = modelARIMA.fit()

## Fit the model
error_Training_ARIMA = mean_squared_error(Y_train, model_fit.fittedvalues)
predicted = model_fit.predict(start = tr_len -1 , end = to_len -1, exog = X_test_ARIMA)[1:]
error_Test_ARIMA = mean_squared_error(Y_test, predicted)
error_Test_ARIMA 

### LSTM Model ###
seq_len = 2 #Length of the seq for the LSTM 

Y_train_LSTM, Y_test_LSTM = np.array(Y_train)[seq_len-1:], np.array(Y_test)
X_train_LSTM = np.zeros((X_train.shape[0]+1-seq_len, seq_len, X_train.shape[1]))
X_test_LSTM = np.zeros((X_test.shape[0], seq_len, X.shape[1]))
for i in range(seq_len):
    X_train_LSTM[:, i, :] = np.array(X_train)[i:X_train.shape[0]+i+1-seq_len, :]
    X_test_LSTM[:, i, :] = np.array(X)\
        [X_train.shape[0]+i-1:X.shape[0]+i+1-seq_len, :]

# LSTM Network #
def create_LSTMmodel(learn_rate = 0.01, momentum= 0):
    # create model 
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train_LSTM.shape[1],X_train_LSTM.shape[2])))
    model.add(Dense(1))
    optimizer = SGD(lr=learn_rate, momentum= 0)
    model.compile(loss='mse', optimizer='adam')
    return model 
LSTMModel = create_LSTMmodel(learn_rate=0.01, momentum=0)
LSTMModel_fit = LSTMModel.fit(X_train_LSTM, Y_train_LSTM, validation_data=(X_test_LSTM, Y_test_LSTM), epochs=330, batch_size=72, verbose=0, shuffle=False)

# Plot the data 
pyplot.plot(LSTMModel_fit.history['loss'], label='train',)
pyplot.plot(LSTMModel_fit.history['val_loss'], '--', label='test',)
pyplot.legend()
pyplot.show()

error_Training_LSTM = mean_squared_error(Y_train_LSTM , LSTMModel.predict(X_train_LSTM))
predicted = LSTMModel.predict(X_test_LSTM)
error_Test_LSTM = mean_squared_error(Y_test, predicted)


test_results.append(error_Test_ARIMA)
test_results.append(error_Test_LSTM)

train_results.append(error_Training_ARIMA)
train_results.append(error_Test_LSTM)

names.append("ARIMA")
names.append("LSTM")


''' Model Tuning and Grid Search '''
def evaluate_arima_model(arima_order):
    #predicted = list()
    modelARIMA=ARIMA(endog=Y_train, exog=X_train_ARIMA, order=arima_order)
    model_fit= modelARIMA.fit()
    error = mean_squared_error(Y_train, model_fit.fittedvalues)
    return error

# Evaluate combinations of p, d and q values for the ARIMA model 
def evaluate_models(p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.7f' % (order,mse))
                except:
                        continue
    print('Best ARIMA%s MSE=%.7f' % (best_cfg, best_score))

# Evaluate parameters 
p_values = [0,1,2]
d_values = range(0,2)
q_values = range(0,2)
evaluate_models(p_values, d_values, q_values)

''' Check the model on the test set '''
#prepare
model_ARIMAtuned = ARIMA(endog=Y_train, exog=X_train_ARIMA, order = [2,0,1])
model_fit_tuned = model_ARIMAtuned.fit()

# estimate accuracy 
predicted_tuned = model_fit.predict(start = tr_len -1 , end = to_len -1, exog = X_test_ARIMA)[1:]
print(mean_squared_error(Y_test, predicted_tuned))

''' Final Graph ''' 
# plotting actual vs predicted
predicted_tuned.index = Y_test.index 
pyplot.plot(np.exp(Y_test).cumprod(), 'r', label = 'actual')

# plotting t, a seperately
pyplot.plot(np.exp(predicted_tuned).cumprod(), 'b--', label = 'predicted')
pyplot.legend()
pyplot.rcParams["figure.figsize"] = (8, 5)
pyplot.show()


