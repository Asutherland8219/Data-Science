# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import datetime
import pandas_datareader as dr
from sklearn.manifold import TSNE
import matplotlib.cm as cm

#Import Model Packages 
from sklearn.cluster import KMeans, AgglomerativeClustering,AffinityPropagation, DBSCAN
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import coint
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_mutual_info_score
from sklearn import cluster, covariance, manifold, metrics


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

drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))

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

# K means model 
distortions = []
max_loop = 20
for k in range (2, max_loop):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)
fig = pyplot.figure(figsize=(15,5))
pyplot.plot(range(2, max_loop), distortions)
pyplot.xticks([ i for i in range(2, max_loop)], rotation=75)
pyplot.grid(True)
pyplot.show()
### With this we are looking for a bend in the graph, or the "elbow". 

## Silhouette score

silhouette_score = []
for k in range (2, max_loop):
    kmeans = KMeans(n_clusters=k, random_state= 20, n_init=10, n_jobs=-1)
    kmeans.fit(X)
    silhouette_score.append(metrics.silhouette_score(X, kmeans.labels_, random_state=10))

fig = pyplot.figure(figsize=(15, 5))
pyplot.plot(range(2, max_loop), silhouette_score)
pyplot.xticks([i for i in range (2, max_loop)], rotation= 75)
pyplot.grid(True)
pyplot.show()

# Our data will be usering 5 clusters vs 6 in the example in  the book / literature
# not enough stocks per cluster so cutting down to 4 clusters


### this route doesnt seem ideal as there is a cluster that is quite small compared to the others 

''' Clustering and visualization of the models '''
nclust = 5

k_means = cluster.KMeans(n_clusters=nclust)
k_means.fit(X)

# Extract labels 
target_labels = k_means.predict(X)

## Visualize in 2d space 
centroids = k_means.cluster_centers_
fig = pyplot.figure(figsize=(16,10))
ax = fig.add_subplot(111)
scatter = ax.scatter(X.iloc[:,0], X.iloc[:, 1], c = k_means.labels_, cmap="rainbow", label = X.index)
ax.set_title('k-means results')
ax.set_xlabel('Mean Return')
ax.set_ylabel('Volatility')
pyplot.colorbar(scatter)
pyplot.plot(centroids[:,0], centroids[:,1], 'sg', markersize=11)

pyplot.show()


# barchart the number of stocks in each cluster
clustered_series = pd.Series(index=X.index, data=k_means.labels_.flatten())
clustered_series_all = pd.Series(index=X.index, data=k_means.labels_.flatten())
clustered_series = clustered_series[clustered_series != -1]

pyplot.figure(figsize=(12, 7))
pyplot.barh(
    range(len(clustered_series.value_counts())), 
    clustered_series.value_counts()
)

pyplot.title('Cluster Member Counts')
pyplot.xlabel('Stocks in Cluster')
pyplot.ylabel('Cluster Number')
pyplot.show()


''' Hierarchical Clustering (Agglomerative Clustering) '''
from scipy.cluster.hierarchy import dendrogram, linkage, ward

z= linkage(X, method='ward')
print(z[0])

pyplot.figure(figsize=(10, 7))
pyplot.title("Stocks Dendrograms")
dendrogram(z, labels= X.index)
pyplot.show()

### Look for the longest vertical distance without any horizontal line passing through it, the number of vertical lines in this newly created hoirzontal line is equal to the number of clusters 
# for this example it was 7 distance threshold

distance_threshold = 12
clusters = fcluster(z, distance_threshold, criterion='distance')
chosen_clusters = pd.DataFrame(data=clusters, columns=['cluster'])
print(chosen_clusters['cluster'].unique())

# n_cluster = 9

### Clustering and visualization 
nclust = 9
hc = AgglomerativeClustering(n_clusters= nclust, affinity= 'euclidean', linkage= 'ward')
clust_labels1 = hc.fit_predict(X)

fig = pyplot.figure(figsize=(16, 10))
ax = fig.add_subplot(111)
scatter = ax.scatter(X.iloc[:,0], X.iloc[:,1], c= clust_labels1, cmap= "rainbow")
ax.set_title('Hierachical Clustering')
ax.set_xlabel('Mean Return')
ax.set_ylabel('volatility')
pyplot.colorbar(scatter)
pyplot.show()

### Affinity propogation method
ap = AffinityPropagation()
ap.fit(X)
clust_labels2 = ap.predict(X)

fig = pyplot.figure(figsize=(10,8))
ax = fig.add_subplot(111)
scatter = ax.scatter(X.iloc[:,0], X.iloc[:,1], c = clust_labels2, cmap = "rainbow")
ax.set_title('Affinity')
ax.set_xlabel('Mean Return')
ax.set_ylabel('Volatility')
pyplot.colorbar(scatter)
pyplot.show()

''' Cluster Evaluation '''
### Higher score = betteer defined clusters
print("km", metrics.silhouette_score(X, k_means.labels_, metric='euclidean'))
print("hc", metrics.silhouette_score(X, hc.fit_predict(X), metric='euclidean'))
print("ap", metrics.silhouette_score(X, ap.labels_, metric='euclidean'))

''' Visualize the return within a Cluster ''' 
# show number of stocks in each cluster
clustered_series_ap = pd.Series(index=X.index, data=ap.labels_.flatten())
# clustered stock with its cluster label
clustered_series_all_ap = pd.Series(index=X.index, data=ap.labels_.flatten())
clustered_series_ap = clustered_series_ap[clustered_series != -1]

clustered_series = pd.Series(index= X.index, data= ap.fit_predict(X).flatten())

clustered_series_all = pd.Series(index= X.index, data=ap.fit_predict(X).flatten())
clustered_series = clustered_series[clustered_series != -1]

counts = clustered_series_ap.value_counts()

cluster_vis_list = list(counts[(counts<25) & (counts>1)].index)[::-1]
print(cluster_vis_list)

CLUSTER_SIZE_LIMIT = 9999
counts = clustered_series.value_counts()
ticker_count_reduced = counts[(counts>1) & (counts<=CLUSTER_SIZE_LIMIT)]
print("Clusteres formed: %d" % len (ticker_count_reduced))
print("Pairs to evaluate: %d" % (ticker_count_reduced * (ticker_count_reduced-1)).sum())

# plot some of the clusters 
pyplot.figure(figsize=(12,7))

print(cluster_vis_list[0:min(len(cluster_vis_list), 4)])

for clust in cluster_vis_list[0:min(len(cluster_vis_list), 4)]:
    tickers = list(clustered_series[clustered_series==clust].index)
    means = np.log(dataset.loc[:"2018-02-01", tickers].mean())
    data = np.log(dataset.loc[:"2018-02-01", tickers]).sub(means)
    data.plot(title="Stock Time Series for Cluster %d" % clust)
pyplot.show()


''' Pairs Selection '''
# cointegration and pair selection function 
def find_cointegrated_pairs(data, significance=0.05):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(1):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue 
            if pvalue < significance:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs

cluster_dict = {}
for i, which_clust in enumerate(ticker_count_reduced.index):
    tickers= clustered_series[clustered_series == which_clust].index
    score_matrix, pvalue_matrix, pairs = find_cointegrated_pairs(dataset[tickers])
    cluster_dict[which_clust] = {}
    cluster_dict[which_clust]['score_matrix'] = score_matrix
    cluster_dict[which_clust]['pvalue_matrix'] = pvalue_matrix
    cluster_dict[which_clust]['pairs'] = pairs
pairs = []

for clust in cluster_dict.keys():
    pairs.extend(cluster_dict[clust]['pairs'])

print("Number of pairs found: %d" % len(pairs))
print("In those pairs, there are %d unique tickers." % len(np.unique(pairs)))

print(pairs)

''' Pair Visualization ''' 

stocks = np.unique(pairs)
X_df = pd.DataFrame(index=X.index, data=X).T

in_pairs_series = clustered_series.loc[stocks]
stocks = list(np.unique(pairs))
X_pairs = X_df.T.loc[stocks]
X_tsne = TSNE(learning_rate=50, perplexity=3, random_state=1337).fit_transform(X_pairs)

pyplot.figure(1, facecolor='white',figsize=(16, 8))
pyplot.clf()
pyplot.axis('off')


for pair in pairs:
    ticker1 = pair[0]
    loc1 = X_pairs.index.get_loc(pair[0])
    x1, y1 = X_tsne[loc1, :]

    ticker2 = pair[0]
    loc2 = X_pairs.index.get_loc(pair[1])
    x2, y2 = X_tsne[loc2, :]

    pyplot.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, c='gray')

pyplot.scatter(X_tsne[:, 0], X_tsne[:, 1], s=220, alpha= 0.9, c= in_pairs_series.values, cmap= cm.Paired)
pyplot.title('T-SNE Visualization of Validated Pairs');

for x, y, name in zip(X_tsne[:,0], X_tsne[:,1], X_pairs.index):

    label = name

    pyplot.annotate(label, (x,y), textcoords="offset points", xytext=(0, 10), ha='center')

pyplot.plot(centroids[:, 0], centroids[:,1], 'sg', markersize= 11)

pyplot.show()

























































