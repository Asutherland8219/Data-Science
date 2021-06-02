# Goal is asset classs allocation 
import pkg_resources
import pip 

# Load libraries
import numpy as np
from numpy import mkl
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from datetime import date

#Import Model Packages 
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from sklearn.metrics import adjusted_mutual_info_score
from sklearn import cluster, covariance, manifold

#Package for optimization of mean variance optimization

dataset = read_csv('Workbook_ex\Datasets\sp500_data.csv', index_col=0)

import warnings
warnings.filterwarnings('ignore')

# if you are installing from a conda env you need to import from conda
from conda import cvxopt as opt 


print(dataset.type)