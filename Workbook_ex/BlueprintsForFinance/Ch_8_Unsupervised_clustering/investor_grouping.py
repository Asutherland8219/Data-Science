# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import datetime

#Import Model Packages 
from sklearn.cluster import KMeans, AgglomerativeClustering,AffinityPropagation
from sklearn.metrics import adjusted_mutual_info_score
from sklearn import metrics
from sklearn import cluster, covariance, manifold


#Other Helper Packages and functions
import matplotlib.ticker as ticker
from itertools import cycle

import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_excel('Workbook_ex\Datasets\ProcessedData.xlsx')
print(type(dataset))
print(dataset.shape)

# peek and describe the data 
set_option('display.width', 100)
set_option('precision', 3)
print(dataset.head(5))
print(dataset.describe())


''' Data Viz ''' 
correlation = dataset.corr()
pyplot.figure(figsize=(15,15))
pyplot.title('Correlation Matrix')
sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
pyplot.show()

''' Data prep and cleaning '''
## Check and remove any null values
print('Null Values =', dataset.isnull().values.any())

X= dataset.copy("deep")
X= X.drop(['ID'], axis= 1)
print(X.head())

''' Evaluate Algorithms and Models '''
distortions = []
max_loop = 40 
for k in range(2, max_loop):
    k_means = KMeans(n_clusters=k)
    k_means.fit(X)
    distortions.append(k_means.inertia_)
fig = pyplot.figure(figsize=(10, 5))
pyplot.plot(range(2, max_loop), distortions)
pyplot.xticks([i for i in range(2, max_loop)], rotation= 75)
pyplot.xlabel("Number of Clusters")
pyplot.ylabel("Sum of Square Error")
pyplot.grid(True)
pyplot.show()

# Silhouette Score 
silhouette_score = []
for k in range(2, max_loop):
    kmeans = KMeans(n_clusters=k, random_state=10, n_init=10, n_jobs=-1)
    kmeans.fit(X)
    silhouette_score.append(metrics.silhouette_score(X, kmeans.labels_, random_state= 10))
fig = pyplot.figure(figsize=(10,5))
pyplot.plot(range(2, max_loop), silhouette_score)
pyplot.xticks([i for i in range(2, max_loop)], rotation= 75)
pyplot.xlabel(" Number of Clusters ")
pyplot.ylabel(" Sum of Square Error ")
pyplot.grid(True)
pyplot.show()

## Based on the graph we created, we came up with the following number of clusters ... 7

nclust = 7

k_means = cluster.KMeans(n_clusters=nclust)
k_means.fit(X)

targe_labels = k_means.predict(X)

ap = AffinityPropagation(damping= 0.5, max_iter= 250, affinity = 'euclidean')
ap.fit(X)
clust_labels2 = ap.predict(X)

cluster_centers_indices = ap.cluster_centers_indices_
labels = ap.labels_
n_clusters_ = len(cluster_centers_indices)
print('Estimated Number of Clusters: %d' % n_clusters_)

# Evaluate Clusters 
print("km", metrics.silhouette_score(X, k_means.labels_, metric='euclidean'))
print("ap", metrics.silhouette_score(X, ap.labels_, metric= 'euclidean'))

''' Cluster intuition ?? '''
cluster_output = pd.concat([pd.DataFrame(X), pd.DataFrame(k_means.labels_, columns= ['cluster'])], axis= 1)
output = cluster_output.groupby('cluster').mean()
print(output)

output[['AGE', 'EDUC', 'MARRIED', 'KIDS', 'LIFECL', 'OCCAT']].plot.bar(rot= 0 , figsize=(18, 5))

output[['HHOUSES', 'NWCAT', 'INCCL', 'WSAVED', 'SPENDMORE', 'RISK']].plot.bar(rot= 0, figsize=(18, 5))

# need to view the graphs in order to come up with a final hypothesis result 




