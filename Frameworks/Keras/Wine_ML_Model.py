import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from keras.models import Sequential
from keras.layers.core import Dense

#read in the data
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

### Pre process the data and begin creating the information for the neural network. This is where we are labeling the data (ie. hot encoding ) and joining the data. This model is based on the relu fuction. The rectified linear unit function: a linear function that will only output if positive. It is the defaul for many neural networks because the model is easier to train and often achieves beter performance.

### Tanh function is another alternative to the ReLu function and stands for the "Hyperbolic Tangent". This function outputs values between -1.0 and 1.0. Tanh is preferred over sigmmoid ###

# add a type column for red with value 1
red['type'] = 1

# add a type column  for white with a value of 0
white['type'] = 0

# Append 'white' to 'red' 
wine = red.append(white, ignore_index=True)

print(wine.head())

# Use the created data to come up with a machine learning model specfically using sklearn

# Specify the data
X = wine.iloc[:,0:11]

# Specify the target labels and flatten the array
y = np.ravel(wine.type)

# Split the data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Standardize the data (Some of the data shows a giant range)

#We must define the scaler with the training set 
scaler = StandardScaler().fit(X_train)

#Scale the training and test set to better fit the data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Layer the model (more data, more layers. In this instance one layer is sufficient)
model = Sequential()

# Add input layer 
model.add(Dense(12, activation='relu', input_shape=(11,)))

# Add one hidden layer
model.add(Dense(8, activation='relu'))

# Add an output layer 
model.add(Dense(1, activation='sigmoid'))

### Time to summarize the model and see how the model was output ###

#Shape the model
model.output_shape

#Model summary  
model.summary()

#Model Config
model.get_config()

# List all weight tensors
model.get_weights()

# Compile and fit the data 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)

## Putting the model to use ##
y_pred = model.predict(X_test)

### Finally score the model and evaluate your results ###
score = model.evaluate(X_test, y_test, verbose=1)

print(score)
## This score is 0.99 or 99%... realistically this is much much higher than you would get from a typical model and in fact may be overfit to the data ##

# Confusion Matrix
## A matrix which shows the correct predictions as well as the incorrect predictions made . in a ideal situation numbers will only be displayed in the diagonal (left to right ) ##
confusion_matrix(y_test, y_pred)

# Precision score
## Precision score is the numeric representative for exactness in the model. Higher is better.
precision_score(y_test, y_pred)

# Recall 
## Recall is a measure of the classifiers completeness. Higher means more cases the classifier covers.
recall_score(y_test, y_pred)

# F1 Score 
## The weighted average of precision and recall 
f1_score(y_test, y_pred)

# Cohen's kappa
## Classification accuracy normalized by the class imbalances in the data 
cohen_kappa_score(y_test, y_pred)