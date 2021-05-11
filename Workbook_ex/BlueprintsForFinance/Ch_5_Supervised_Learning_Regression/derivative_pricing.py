''' Function and modules for the Supervised Regression Models '''
# import norm function 
from scipy.stats import norm 

# Hide the warnings
import warnings 
warnings.filterwarnings('ignore')

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

''' Using the Black Scholes model for option price calculations '''
''' In this model we use variables such as :
                - Stock price   ( S )
                - Strike Price  ( K )
                - Time to Expiration ( T-t )
                - Volatility (Sigma)
                - Interest Rate ( r )
                - Dividend yield  ( q )
'''

true_alpha = 0.1
true_beta = 0.1
true_sigma0 = 0.2

risk_free_rate = 0.05

''' Volatitily and option pricing functions '''
# We create data points for this for simplicity 
def option_vol_from_surface(moneyness, time_to_maturity):
    return true_sigma0 + true_alpha * time_to_maturity + true_beta * np.square(moneyness - 1 )

def call_option_price(moneyness, time_to_maturity, option_vol):
    d1=(np.log(1/moneyness)+(risk_free_rate + np.square(option_vol))*time_to_maturity)/(option_vol * np.sqrt(time_to_maturity))
    d2=(np.log(1/moneyness)+(risk_free_rate + np.square(option_vol))*time_to_maturity)/(option_vol * np.sqrt(time_to_maturity))
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)

    return N_d1 - moneyness * np.exp(-risk_free_rate*time_to_maturity) * N_d2
    
# Generate the data points 
N = 10000

Ks= 1+0.25*np.random.randn(N)
Ts = np.random.random(N)
Sigmas = np.array([option_vol_from_surface(k,t) for k,t in zip(Ks, Ts)])
Ps = np.array([call_option_price(k,t,sig) for k,t,sig in zip(Ks,Ts,Sigmas)])

Y = Ps
X = np.concatenate([Ks.reshape(-1,1), Ts.reshape(-1,1), Sigmas.reshape(-1,1)], axis=1)

dataset= pd.DataFrame(np.concatenate([Y.reshape(-1, 1), X], axis=1), columns=['Price', 'Moneyness', 'Time', 'Vol'])
print(dataset.head())
# Data visualization 

scatter_matrix(dataset, figsize=(12,12))
pyplot.show()

# Univariation feature selections 
bestfeatures = SelectKBest(k='all', score_func=f_regression)
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(['Moneyness','Time', 'Vol'])
# concat two df for better visualization 
featureScores = pd.concat([dfcolumns,dfscores], axis = 1)
featureScores.columns = ['Specs', 'Score'] # naming the data frame columns 
featureScores.nlargest(10, 'Score').set_index('Specs')

''' Train, test and split the models ''' 
validation_size = 0.2 
train_size = int(len(X) * (1-validation_size))
X_train, X_test = X[0:train_size], X[train_size:len(X)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(X)]

# k fold parameters
num_folds = 10 
seed = 7
scoring = 'neg_mean_squared_error'


# compare the algorithms
models = []

# linear models 
models.append(('LR', LinearRegression()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

# Artificial Neurla Networks
models.append(('MLP', MLPRegressor()))

# Boosting and Bagging methods 
## Boosting 
models.append(('ABR', AdaBoostRegressor()))
models.append(('GBR', GradientBoostingRegressor()))

## Bagging methods 
models.append(('RFR', RandomForestRegressor()))
models.append(('ETR', ExtraTreesRegressor()))

''' Model Tuning and Finalizing the model ''' 
# too many middle nodes in middle layer can create a too well fitted model and it wont generate data 

'''
hidden_layer_sizes : tuple, length = n_layers - 2, default(100,)
The ith element represents the number of neurons in the ith hidden layer. 
'''

param_grid= {'hidden_layer_sizes': [(20,), (50,), (20,20), (20,30,20)]}
model = MLPRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X_train, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

## this helps us identify the best possible model  combo with the ideal hidden layer sizes
## now we can prepare the model 

model_tuned = MLPRegressor(hidden_layer_sizes=(20, 30, 20))
model_tuned.fit((X_train, Y_train))

# estimate accuracy and transform the validation data set 
predictions = model_tuned.predict(X_test)
print(mean_squared_error(Y_test, predictions))

## note: the mean_squared_erro in this instance is actually RMSE

''' Additional analysis of the data; removing the volatitlity '''
# we are now trying to predict the price without the volatility data 
## we are diting the data and recreating the training and test size 
'''
X = [:, :2]
validation_size = 0.2
train_size = int(len(X) * (1-validation_size))
X_train, X_test = X[0:train_size], X[train_size:len(X)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(X)]
'''