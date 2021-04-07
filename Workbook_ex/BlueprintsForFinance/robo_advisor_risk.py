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
import matplotlib as plt
import copy
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
from statsmodels.graphics.tsaplots import plot_acf

# import to save the model
from pickle import dump
from pickle import load 

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
pyplot.show()

sentiment09 = sns.histplot(dataset['RT09'], kde=False, bins=int(180/5), color= 'blue', line_kws={'edgecolor':'black'})
pyplot.show()

## narrow down the dataset to pick out the aforementioned "intelligent investors"

dataset3 = copy.deepcopy(dataset)
dataset3 = dataset[dataset['PercentageChange']<=.1]
dataset3['TrueRiskTolerance'] = (dataset3['RT07'] + dataset3['RT09'])/2

dataset3.drop(labels=['RT07', 'RT09'], axis=1, inplace=True)
dataset3.drop(labels=['PercentageChange'], axis=1, inplace=True)

### With this data set there are 500+ features, we must narrow down that list. 
keep_list2= ['AGE07', 'EDCL07', 'MARRIED07', 'KIDS07', 'OCCAT107', 'INCOME07', 'RISK07', 'NETWORTH07', 'TrueRiskTolerance']

drop_list2= [col for col in dataset3.columns if col not in keep_list2]

dataset3.drop(labels=drop_list2, axis=1, inplace=True)

### Create a correlation graph with these features 
correlation = dataset3.corr()
pyplot.figure(figsize=(15,15))
pyplot.title('Correlation Matrix')
sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')

pyplot.show()

''' Evaluate the models and decide which is ideal for this scenario ''' 
# Train-Test Split 
Y = dataset3["TrueRiskTolerance"]
X = dataset3.loc[:, dataset3.columns != 'TrueRiskTolerance']
validation_size = 0.2 
seed = 3
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

num_folds = 10
scoring = 'r2'

# Model Selection 
## Regression and tree regression algorithms 
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

## Ensemble Models 

### Boosting Methods 
models.append(('ABR', AdaBoostRegressor()))
models.append(('GBR',GradientBoostingRegressor()))

### Bagging Methods 
models.append(('RFR', RandomForestRegressor()))
models.append(('ETR', ExtraTreesRegressor()))

names = []
results = []

for name, model in models :

    ## K-Fold Analysis
    kfold = KFold(n_splits=num_folds, random_state=seed)
    ## Convert MSE to positive, lower = better
    cv_results = -1*cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name,cv_results.mean(), cv_results.std())
    print(msg)
### Compare CV results 
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
fig.set_size_inches(15,8)
pyplot.show()

''' Model Tuning and Grid search'''
# for this we have chosen the random forest regressor model 
'''
n_estimators : integer, optional (default=10)
'''
param_grid = {'n_estimators': [50,100,150,200,250,300,350,400]}
model = RandomForestRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X_train, Y_train)
print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']


model = RandomForestRegressor(n_estimators= 200, n_jobs=-1)
model.fit(X_train,Y_train)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
pyplot.show()

''' Save model for later use '''
filename = 'finalized_model.sav'
dump(model, open(filname, 'wb'))

#load the model 
loaded_model = load(open(filname, 'rb'))

#estimate accuracy
predictions = loaded_model.predict(X_validation)
result = mean_squared_error(Y_validation, predictions)
print(r2_score(Y_validation, predictions))
print(result)
