''' Taking a pre labeled set and then using our own created clusters to analyze '''
# note the iris set is super small, with only 150 points and 4 features. Limited availability in terms of analysis

# need matplotlib to show graphs
import matplotlib.pyplot as pyplot
from seaborn.external.husl import f

from sklearn.datasets import load_iris
iris = load_iris()

print(iris.DESCR)

print(iris.target_names)
print(iris.feature_names)

''' Explode the data set further '''
import pandas as pd 
pd.set_option('max_columns', 5)
pd.set_option('display.width', None)

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

iris_df['species'] = [iris.target_names[i] for i in iris.target]

print(iris_df.head())

pd.set_option('precision', 2)

# peek the new column and data set 
print(iris_df.describe)

# peek the column itself 
print(iris_df['species'].describe())

''' visualize the new data set '''
import seaborn as sns 
sns.set(font_scale=1.1)
sns.set_style('whitegrid')
grid = sns.pairplot(data=iris_df, vars=iris_df.columns[0:4], hue='species')
pyplot.show()

''' Using Kmeans estimator for clustering '''
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=11)

kmeans.fit(iris.data)

'''' Dimensionality reduction with PCA '''
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=11)

pca.fit(iris.data)
iris_pca = pca.transform(iris.data)


# visualize the reduced data 
iris_pca_df = pd.DataFrame(iris_pca, columns=['Component1', 'Component2'])
iris_pca_df['species'] = iris_df.species

axes = sns.scatterplot(data=iris_pca_df, x = 'Component1', y = 'Component2', hue = 'species', legend= 'brief', palette= 'cool')

iris_centers = pca.transform(kmeans.cluster_centers_)

dots = pyplot.scatter(iris_centers[:, 0], iris_centers[:, 1], s= 100, c='k')

''' Choose the best clustering estimator  to use '''
from sklearn.cluster import DBSCAN, MeanShift, SpectralClustering, AgglomerativeClustering

estimators = {
    'KMeans': kmeans,
    'DBSCAN': DBSCAN(),
    'MeanShift': MeanShift(),
    'SpectralClustering': SpectralClustering(n_clusters=3),
    'AgglomerativeClustering': AgglomerativeClustering(n_clusters=3)
}

import numpy as np 

for name, estimator in estimators.items():
    estimator.fit(iris.data)
    print(f'\n{name}:')
    for i in range(0, 101, 50):
        labels, counts = np.unique(
            estimator.labels_[i:i+50], return_counts=True
        )
        print(f'{i} - {i+50}:')
        for label, count in zip(labels, counts):
            print(f' label={label}, count={count}')

