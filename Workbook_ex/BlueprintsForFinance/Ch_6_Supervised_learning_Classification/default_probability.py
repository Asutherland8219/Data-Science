## 1 for charge off , 0 for other wise
# load the data set (which is frking hugeee) can be found here : https://oreil.lyDG9j5


''' Determining whether a transaction is fraudulent or not  '''

''' Function and modules for the Supervised learning Classification models '''
from pdb import Pdb
from re import VERBOSE
from pandas.core.arrays import categorical
from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
from sklearn.model_selection import train_test_split, KFold, cross_val_score , GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.ops.array_ops import _cast_nested_seqs_to_dtype

''' Function and modules for Data Analysis and Model Evaluation '''
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
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
from pandas import read_csv, set_option
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
from statsmodels.graphics.tsaplots import plot_acf


# import to save the model
from pickle import dump
from pickle import load 



dataset = pd.read_csv('Data-Science\Workbook_ex\BlueprintsForFinance\Ch_6_Supervised_learning_Classification\Loans_data.gz', compression='gzip', low_memory=True)

print(dataset.shape)
print(dataset.head())                      

dataset['loan_status'].value_counts(dropna=False)

dataset = dataset.loc[dataset['loan_status'].isin(['Fully Paid', 'Charged Off'])]
dataset['loan_status'].value_counts(normalize=True, dropna=False)

## Now we have to create a binary column in order to classify each as paid off or charge off 
dataset['charged_off'] = (dataset['loan_status'] == 'Charged Off').apply(np.uint8)
dataset.drop('loan_status', axis=1, inplace=True)

## Limit the feature space, we have over 150 features in this data set... way too many. We need to eliminate features with little to no importance 

'''1. Eliminate missing values greater than 30% (I think this is high in a data set this large, 20% might be more apt for smaller data sets.)'''
missing_fractions = dataset.isnull().mean().sort_values(ascending=False)
drop_list = sorted (list(missing_fractions[missing_fractions > 0.3].index))
dataset.drop(labels=drop_list, axis=1, inplace=True)
print(dataset.shape)

''' 2. Elminate features that arent pertinent to FICO scores '''
keep_list = ['charged_off', 'funded_amnt', 'addr_state', 'annual_inc', 'application_type', 'dti', 'earlest_cr_line', 'emp_length', 'emp_title', 'fico_range_high', 'fico_range_low', 'grade', 'home_ownership', 'id', 'initial_list_status','installment', 'int_rate', 'loan_amnt', 'loan_status', 'mort_acc', 'open_acc', 'pub_rec' \
    'pub_rec_bankruptcies', 'purpose', 'revol_bal', 'revol_util', 'sub_grade', 'term', 'title', 'total_acc', 'verification_status', 'zip_code', 'last_pymnt_amnt', 'num_actv_rev_tl', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_old_rev_tl_op', 'bc_util', 'bc_open_to_buy', 'avg_cur_bal', 'acc_open_past_24mths']

drop_list = [col for col in dataset.columns if col not in keep_list]
dataset.drop(labels=drop_list, axis=1, inplace=True)
print(dataset.shape)

''' 3. Eliminate feature based on correlation '''
# this is probably the one to be the most careful with, we dont want to drop a useful variable
correlation = dataset.corr()
correlation_chargeOff = abs(correlation['charged_off'])
drop_list_corr = sorted(list(correlation_chargeOff[correlation_chargeOff < 0.03].index))
print(drop_list_corr)

# drop the strongly correlated columns 
dataset.drop(labels=drop_list_corr, axis=1, inplace=True)

print(dataset.head())

''' EDA to figure out if the data should be removed '''

dataset.drop(['id','emp_title','title','zip_code'], axis=1, inplace=True)

dataset['term'] = dataset['term'].apply(lambda s: np.int8(s.split()[0]))
print(dataset.groupby('term')['charged_off'].value_counts(normalize=True).loc[:, 1])

## short term loans charge off sooner that long term 
dataset['emp_length'].replace(to_replace='10+ years', value='10 years', inplace=True)
dataset['emp_length'].replace('< 1 year', '0 years', inplace=True)

def emp_length_to_int(s):
    if pd.isnull(s):
        return s
    else: 
        return np.int8(s.split()[0])

dataset['emp_length'] = dataset['emp_length'].apply(emp_length_to_int)
charge_off_rates = dataset.groupby('emp_length')['charged_off'].value_counts(normalize=True).loc[:,1]
sns.barplot(x=charge_off_rates.index, y=charge_off_rates.values)
pyplot.show()

dataset.drop(['emp_length'], axis=1, inplace=True)

charge_off_rates = dataset.groupby('sub_grade')['charged_off'].value_counts(normalize=True).loc[:, 1]
sns.barplot(x= charge_off_rates.index, y=charge_off_rates.values)
pyplot.show()

## now we check annual income for info
dataset[['annual_inc']].describe()
dataset[['log_annual_inc']] = dataset[['annual_inc']].apply(lambda x: np.log10(x+1))
dataset.drop('annual_inc', axis=1, inplace=True)

print(dataset[['fico_range_low', 'fico_range_high']].corr())

## 1/1 correlation so mayswell only keep one 
dataset['fico_score'] = 0.5*dataset['fico_range_low'] + 0.5*dataset['fico_range_high']

dataset.drop(['fico_range_high', 'fico_range_low'], axis=1, inplace=True)

''' Convert data to numerical '''
## some data is in letter or string format, it needs to be changed to numerical data to be useable in a model 
## this will be done using hot label encoding 

from sklearn.preprocessing import LabelEncoder
categorical_feature_mask = dataset.dtypes==object

categorical_cols = dataset.columns[categorical_feature_mask].tolist()

print(categorical_cols)

datatypeseries = dataset.dtypes
print('Data Type of each column in Dataframe :')
print(datatypeseries)

# now we convert with the encoder 
le = LabelEncoder()

dataset[categorical_cols]= dataset[categorical_cols].apply(lambda col: le.fit_transform(col))
print(dataset[categorical_cols].head())

''' Sampling the data ''' 
loanstatus_0 = dataset[dataset["charged_off"]==0]
loanstatus_1 = dataset[dataset["charged_off"]==1]
subset_of_loanstatus_0 = loanstatus_0.sample(n=5500)
subset_of_loanstatus_1 = loanstatus_1.sample(n=5500)
dataset = pd.concat([subset_of_loanstatus_1, subset_of_loanstatus_0])
dataset= dataset.sample(frac=1).reset_index(drop=True)
print("Current shape of dataset :", dataset.shape)

values = dataset.mean()

# clean up the NA wiht mean of column
dataset.fillna(value=values, inplace=True)
 
print(dataset.head())


''' Train test split '''
Y = dataset['charged_off']
X = dataset.loc[:, dataset.columns != 'charged off']
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y, test_size=validation_size, random_state=seed)
num_folds = 10
scoring='roc_auc'

''' Evaluate Algo and models ''' 

# spot check the algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#Neural Network
models.append(('NN', MLPClassifier()))
#Ensable Models 
# Boosting methods
models.append(('AB', AdaBoostClassifier()))
models.append(('GBM', GradientBoostingClassifier()))
# Bagging methods
models.append(('RF', RandomForestClassifier()))
models.append(('ET', ExtraTreesClassifier()))

results = []
names = [] 
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# graph it 
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
fig.set_size_inches(15, 8)
pyplot.show()















