import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


#read in the data
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

# Pre process the data and begin creating the information for the neural network. This is where we are labeling the data (ie. hot encoding ) and joining the data

# add a type column for red with value 1
red['type'] = 1

# add a type column  for white with a value of 0
white['type'] = 0

# Append 'white' to 'red' 
wine = red.append(white, ignore_index=True)

print(wine.head())

#Create a correlation matrix using seaborn
corr = wine.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
plt.show()

