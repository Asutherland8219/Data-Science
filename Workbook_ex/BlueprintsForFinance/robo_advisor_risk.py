''' Modeling investor Risk Tolerance and enabling a ML ROBO advisor '''

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

# load the dataset
dataset = pd.read_excel('Data-Science\Workbook_ex\BlueprintsForFinance\SCFP2009panel.xlsx')

print(dataset.shape) ### first number is the X or the rows, Y is the columns. The y (515) is also the number of features

### We now have to test the risk tolerance

# compute risky and risk free assets for the year 2007
dataset['RiskFree07']= dataset['LIQ07'] + dataset['CDS07'] + dataset['SAVBND07'] + dataset['CASHLI07']
dataset['Risky07']= dataset['NMMF07'] + dataset['STOCKS07'] + dataset['BOND07']

# compute risky and reisk free assets for the year 2009
dataset['RiskFree09']= dataset['LIQ09'] + dataset['CDS09'] + dataset['SAVBND09'] + dataset['CASHLI09']
dataset['Risky09']= dataset['NMMF09'] + dataset['STOCKS09'] + dataset['BOND09']

# computer the risk tolerance for 2007
dataset['RT07']= dataset['Risky07']/(dataset['Risky07']+dataset['RiskFree07'])

# Average stock index to normalize the data from the risky assets in 2009
Average_SP500_2007=1478
Average_SP500_2009=948

# compute data risk tolerance for 2009
dataset['RT09'] = dataset['Risky09']/(dataset['Risky09']+dataset['RiskFree09'])*(Average_SP500_2009/Average_SP500_2007)

print(dataset.head())

## computer percent change in the data set
dataset['PercentageChange'] = np.abs(dataset['RT09']/dataset['RT07']-1)

## Drop NA and NAN rows
dataset = dataset.dropna(axis=0)

dataset=dataset[~dataset.isin([np.nan, np.inf, -np.inf]).any(1)]

## Plot and visualize the data 
sentiment07 = sns.histplot(dataset['RT07'], kde=False, bins=int(180/5), color= 'blue', line_kws={'edgecolor':'black'})

sentiment09 = sns.histplot(dataset['RT09'], kde=False, bins=int(180/5), color= 'blue', line_kws={'edgecolor':'black'})


''' Graphs keep overlapping, needs further analysis ''' 




