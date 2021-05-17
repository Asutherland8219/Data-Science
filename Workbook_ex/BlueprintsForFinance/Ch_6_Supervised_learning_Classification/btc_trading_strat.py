# this one is important to pay attention too as it pertains to the crypto model i will be building 
# Load libraries
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
dataset = pd.read_csv('Data-Science\Workbook_ex\BlueprintsForFinance\Ch_6_Supervised_learning_Classification\Btc_data.zip', compression='zip', low_memory=True)

print(dataset.head())

## A lot of the data is NaN which is annoying to work with, we should drop it, data super skewed otherwise


dataset = dataset.dropna()

## dropping timestamp because time does not really matter in this instance, we have the data by the minute 

dataset= dataset.drop(columns=['Timestamp'])

print(dataset.head())
print(dataset.shape)
print(dataset.describe())



