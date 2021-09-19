### Trying out another model; Support Vector Machines ###
### Machine learning tutorial on Sci Kit learn from Data camp; tutorial can be found here : https://www.datacamp.com/community/tutorials/machine-learning-python ###
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import numpy as np

# Preprocessing tools
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV

# Visualization tools 
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt

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

# Split the data into training and test sets; we can evaluate the best metrics for the model as follows:
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.5, random_state=0)

# Set the parameter candidates
parameter_candidates = [
    {'C' : [1, 10, 100, 1000], 'kernel' : ['linear']},
    {'C' : [1, 10, 100, 1000], 'gamma' : [0.001, 0.0001], 'kernel' : ['rbf']},
]

# Creat a classifier with the parameter candidates we just created
clk = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)

# Train the classifier with the training data
clk.fit(X_train, y_train)

# Print ouf THE results 
print('Best score for training data:', clk.best_score_)
print('Best `C`:', clk.best_estimator_.C)
print('Best Kernel:', clk.best_estimator_.kernel)
print('Best `gamma`:', clk.best_estimator_.gamma)

X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(digits.data, digits.target, digits.images, test_size=0.25, random_state=42)

# Create the SVC Model 
svc_model = svm.SVC(gamma=0.001, C=100., kernel='linear')

# Fit the data to the svc model 
svc_model.fit(X_train, y_train)

# Apply classifier to the test data and view the accuracy score
clk.score(X_test, y_test)

# Train and score a new classifier with the grid serach parameters; this is based on the results we calculated above
svm.SVC(C=10, kernel='rbf', gamma=0.001).fit(X_train, y_train).score(X_test, y_test)
# Kernel is a similarity function; used to computer similarity between data points. A kernel added to an algorithm with labels and training data would create a classifier.
# SVM typically divide data points linearly

# Predict the label of 'X_test'
print(svc_model.predict(X_test))

# Predicted values
predicted = svc_model.predict(X_test)

# Print 'y_test' to check the results
print(y_test)

## How does the model perform? ##
# Classification report of 'y_test' and 'predicted'
print(metrics.classification_report(y_test, predicted))

# Confusion matrix
print(metrics.confusion_matrix(y_test, predicted))

# Visualize the SVM with the Isomap() function
# Create an Isomap and fit the 'digits' data to it
X_iso = Isomap(n_neighbors=10).fit_transform(X_train)

# Computer cluster centers and predict cluster index for each sample 
predicted = svc_model.predict(X_train)

# Create a plot with subplots in a grid of 1X2
fig, ax = plt.subplots(1, 2, figsize=(8,4))

# Adjust the layout
fig.subplots_adjust(top=0.85)

# Add scatterplots to the subplots
ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=predicted)
ax[0].set_title('Predicted labels')
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train)
ax[1].set_title('Actual labels')

# Add title
fig.suptitle('Predicted versus actual labels', fontsize=14, fontweight='bold')

# Show the plot 
plt.show()