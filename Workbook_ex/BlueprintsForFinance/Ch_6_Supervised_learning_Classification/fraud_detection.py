## Dataset can be found at https://oreil.ly/CeFRs
## in this model, class or the classification model is binary, 1 if fraudulant and 0 if otherwise 

''' Determining whether a transaction is fraudulent or not  '''

''' Function and modules for the Supervised learning Classification models '''
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

''' Function and modules for Data Analysis and Model Evaluation '''
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
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

dataset = pd.read_csv('Data-Science\Workbook_ex\BlueprintsForFinance\Ch_6_Supervised_learning_Classification\creditcard.csv')
print(dataset.shape)

set_option('display.width', 100)
print(dataset.head(5))