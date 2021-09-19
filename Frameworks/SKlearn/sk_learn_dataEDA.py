### Machine learning tutorial on Sci Kit learn from Data camp; tutorial can be found here : https://www.datacamp.com/community/tutorials/machine-learning-python ###
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Preprocessing tools
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

# Learning model tools
from sklearn import cluster

# Evaluation Tools
from sklearn import metrics
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score

# load the data 
digits = datasets.load_digits()

# Display the data set 
print(digits)

# Explore the data set 
print(digits.keys())

# Print data 
print(digits.data)

# Print out the target values
print(digits.target)

# Print ouf the description
print(digits.DESCR)

## Because this data is numpy array; the data is not sorted in an excel sheet format. It is important to know about the shape of the array to accurately handle the data. ##
# WE must seperate the data into a more usuable format #

# Isolate the digits data 
digits_data = digits.data

# Inspect
print(digits_data.shape)

# Isolate target values with the function target
digits_target = digits.target

# Inspect the shape of the new variable
print(digits_target.shape)

# Print the number of unique labels 
number_digits = len(np.unique(digits.target))

# Isolate the 'images' variable
digits_images = digits.images

# Inspect the shape of the digits set
print(digits_images.shape)

# You notice in the results that in the images set they are 8px by 8px. However, the target data has 64 features according to our inspection. We can reshape the data in order to visualize it. #
print(np.all(digits.images.reshape((1797,64)) == digits.data))

### Now we can vizualize the data; using Matplotlib ###

# Create the figure size you want (width, height) in inches
fig = plt.figure(figsize=(6, 6))

# Adjusdt the subplots 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

## For each of the 64 images created we modify the following ##
for i in range(64):
    
    # Initialize the subplots: add a subplot in the grid of 8 by 8 at the i+1-th position
    ax = fig.add_subplot(8,8, i + 1, xticks=[], yticks=[])

    # Display an image at the i-th position
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

    # Label the image with the target value 
    ax.text(0,7, str(digits.target[i]))

# Show the master plot
plt.show()

### Pre process the data in preperation for the model ###

# Apply 'scale()' to the 'digits' data in order to normalize and scale the data and making each attribute have a mean of zero and a SD of 1
data = scale(digits.data)

# Split into a training set and a test set for the model; this is typically done in a 2/3 training and 1/3 test set #
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data, digits.target, digits.images, test_size=0.25, random_state=42)

# Inspect the data before creating the model
n_samples, n_features = X_train.shape

print(n_samples)

print(n_features)

# Number of training labels 
n_digits = len(np.unique(y_train))

# Inspect 'y_train'
print(len(y_train))

### Begin the learning model ###
## The number of clusters (n_clusters) is the number of groups you want the data to form and the centroids to generate ##
clf = cluster.KMeans(init='k-means++', n_clusters=10)

clf.fit(X_train)

# Next we predict the labels for X_test
y_pred = clf.predict(X_test)

# Verify the data 
print(len(y_pred))
print(len(y_test))

print(y_pred[:100])
print(y_test[:100])

# Check the shape of the cluster centers
clf.cluster_centers_.shape

## Evaluate the model ##
print(metrics.confusion_matrix(y_test, y_pred))

print('% 9s' % 'inertia  homo  compl  v-meas  ARI   AMI  silhoutte')
print('%i  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f' % (clf.inertia_, homogeneity_score(y_test, y_pred), completeness_score(y_test, y_pred), v_measure_score(y_test, y_pred), adjusted_rand_score(y_test,y_pred), adjusted_mutual_info_score(y_test, y_pred), silhouette_score(X_test, y_pred, metric='euclidean')))

# Homogeniety score; what extent all of the clusters contain only data points which are members of a single class (Higher is better)

# Completeness score; how close members of a given class are elements of the same cluster (Higher is better)

# V-measure; harmonic mean between homogeity and completeness 

# Adjusted Rand score; measures the similiarity between two clustering and considers all pairs that are assigned to the same or different clusters 

# Adjusted Mutual Info (AMI); used to compare clusters, measures the similariy between the data points. A value of 1 is when clusterings are equivalent. Normalized against chance

# Silhoutte Score; measures how slimilar an object is to its own cluster compared to other clusters. Ranges from -1 to 1; higher indicates matches to its own cluster and lower  to its neighboring cluster. Scores around zero (0) indicate overlapping clusters, higher indicates lots of seperation. A negative value would indicate that it has been assigned to the wrong cluster







