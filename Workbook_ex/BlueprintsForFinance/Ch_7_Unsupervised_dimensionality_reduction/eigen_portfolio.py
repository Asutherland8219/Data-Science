import numpy as np 
import pandas as pd 
import matplotlib.pyplot as pyplot
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# model packages
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from numpy.linalg import inv, eig, svd

from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA

# load data set 
dataset= read_csv('Workbook_ex\Datasets\Dow_adjcloses.csv', index_col=0)

print(dataset.head())

print(dataset.shape)

correlation = dataset.corr()
pyplot.figure(figsize=(15, 15))
pyplot.title('Correlation Matrix')
sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
pyplot.show()


# Describe data
set_option('precision', 3)
print(dataset.describe())

''' Data Cleaning '''
missing_fractions = dataset.isnull().mean().sort_values(ascending=False)
print(missing_fractions.head(10))

drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
dataset.drop(labels=drop_list, axis=1, inplace=True)
print(dataset.shape)

#fill the data with ffill
dataset= dataset.fillna(method='ffill')
