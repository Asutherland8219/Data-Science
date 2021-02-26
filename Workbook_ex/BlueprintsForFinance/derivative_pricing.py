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
true_sigma = 0.2

risk_free_rate = 0.05

''' Volatitily and option pricing functions '''
def option_vol_from_surface(moneyness, time_to_maturity):
    return true_sigma0 + true_alpha * time_to_maturity + true_beta * np.square(moneyness - 1 )

def call_option_price(moneyness, time_to_maturity, option_vol):
    d1=(np.log(1/moneyness)+(risk_free_rate + np.square(option_vol))*time_to_maturity)/(options_vol * np.sqrt(time_to_maturity))
    d2=(np.log(1/moneyness)+(risk_free_rate + np.square(option_vol))*time_to_maturity)/(options_vol * np.sqrt(time_to_maturity))
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)

    return N_d1 - moneyness * np.exp(-risk_free_rate*time_to_maturity) * N_d2
    
    

