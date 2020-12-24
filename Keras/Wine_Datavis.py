import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#read in the data
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

#create the graph and visualization for the data
fig, ax = plt.subplots(1,2)

ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label='Red Wine')
ax[1].hist(white.alcohol, 10, facecolor='white', ec='black', lw=0.5, alpha=0.5, label='White Wine')

fig.subplots_adjust(left=0.2, right=0.5, bottom=0.2, top=0.5, hspace=0.05, wspace=0.5)
ax[0].set_ylim([0,1000])
ax[0].set_xlabel("Alcohol in % Vol")
ax[0].set_ylabel("Frequency")
ax[1].set_xlabel("Alcohol in % Vol")
ax[1].set_ylabel("Frequency")

fig.suptitle("Distribution of Alcohol in % Vol")

plt.show()

#histogram as an alternative visual
print(np.histogram(red.alcohol, bins=[7,8,9,10,11,12,13,14,15]))

print(np.histogram(white.alcohol, bins=[7,8,9,10,11,12,13,14,15]))

#visualize the Sulfates of the wine in a scatter plot format
fig, ax = plt.subplots(1,2,figsize=(8,4))

ax[0].scatter(red['quality'], red['sulphates'],color='red')
ax[1].scatter(white['quality'], white['sulphates'], color='white', edgecolors="black", lw=0.5)

ax[0].set_title("Red Wine")
ax[1].set_title("White Wine")
ax[0].set_xlabel("Quality")
ax[1].set_xlabel("Quality")
ax[0].set_ylabel("Sulphates")
ax[1].set_ylabel("Sulphates")
ax[0].set_xlim([0,10])
ax[1].set_xlim([0,10])
ax[0].set_ylim([0,2.5])
ax[1].set_ylim([0,2.5])
fig.subplots_adjust(wspace=0.5)
fig.suptitle("Wine Quality by Amount of Sulphates")

plt.show()

#This scatter plot shows that the red wines tend to have more sulphates than the white

#Acidity graph  
#random selection of sample data 
np.random.seed(570)

redlabels = np.unique(red['quality'])
whitelabels = np.unique(white['quality'])

#figure of scatterplot  using different colors as identifiers
fig, ax = plt.subplots(1,2, figsize= (8,4))
redcolors = np.random.rand(6,4)
whitecolors = np.append(redcolors, np.random.rand(1,4), axis=0)

for i in range(len(redcolors)):
    redy = red['alcohol'][red.quality == redlabels[i]]
    redx = red['volatile acidity'][red.quality == redlabels[i]]
    ax[0].scatter(redx, redy, c=redcolors[i])

for i in range(len(whitecolors)):
    whitey = white['alcohol'][white.quality == whitelabels[i]]
    whitex = white['volatile acidity'][white.quality == whitelabels[i]]
    ax[1].scatter(whitex, whitey, c=whitecolors[i])

ax[0].set_title("Red Wine")
ax[1].set_title("White Wine")
ax[0].set_xlabel("Volatile Acidity")
ax[1].set_xlabel("Volatile Acidity")
ax[0].set_ylabel("Alcohol")
ax[1].set_ylabel("Alcohol")
ax[0].set_xlim([0,1.7])
ax[1].set_xlim([0,1.7])
ax[0].set_ylim([5,15.5])
ax[1].set_ylim([5,15.5])

ax[1].legend(whitelabels, loc='best', bbox_to_anchor=(1.3,1))

fig.subplots_adjust(top=0.85, wspace=0.7)

plt.show()


