# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import datetime
import pandas_datareader as dr

#Import Model Packages 
from sklearn.cluster import KMeans, AgglomerativeClustering,AffinityPropagation, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_mutual_info_score
from sklearn import cluster, covariance, manifold


#Other Helper Packages and functions
import matplotlib.ticker as ticker
from itertools import cycle

# disable warnings
import warnings
warnings.filterwarnings('ignore')

# load the dataset we created 
dataset = read_csv('Workbook_ex\Datasets\sp500_data.csv', index_col=0)

print(type(dataset), dataset.shape)

set_option('display.width', 100)
print(dataset.head(5))

set_option('precision', 3)
print(dataset.describe())

''' Data Cleaning ''' 
# check for null values and remove them 
print('Null values =', dataset.isnull().values.any())

missing_fractions = dataset.isnull().mean().sort_values(ascending=False)

print(missing_fractions.head())

drop_list = sorted(list(missing_fractions[missing_fractions > 0.4].index))

dataset.drop(labels=drop_list, axis=1, inplace=True)

print(dataset.shape)

# filling the rows that we dropped

dataset = dataset.fillna(method='ffill')
print(dataset.head())

''' Data Transformation ''' 
returns = dataset.pct_change().mean() * 252
returns = pd.DataFrame(returns)

returns.columns = ['Returns']
returns['Volatility'] = dataset.pct_change().std() * np.sqrt(252)

data= returns

scaler = StandardScaler().fit(data)
rescaledDataset = pd.DataFrame(scaler.fit_transform(data), columns= data.columns, index= data.index)

print(rescaledDataset.head(2))

X = rescaledDataset
print(X.head(2))

''' Evaluate Algorithms and models '''
