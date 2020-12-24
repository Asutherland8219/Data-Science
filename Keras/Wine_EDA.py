import pandas as pd

#read in the data
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

#print the tops of the data to make sure it is read in correctly

print(white.head())
print(red.head())

#print out the info of the data in order to get a better idea of what types of data we are working with 

print(white.info())
print(red.info())

#check data types and also make sure that the data is not null 

print(white.describe())
print(red.describe())

#null
pd.isnull(red)
pd.isnull(white)


