from scipy.sparse.construct import random
from sklearn.datasets import load_digits
from sklearn.utils import shuffle
digits = load_digits

''' Explore the data '''
print(digits.data.shape)
print(digits.target.shape)

## The data is the images while the targets are the labels

''' Visualize the data'''
import matplotlib as pyplot

figure, axes = pyplot.subplots(nrows=4, ncols=6, figsize=(6,4))
for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    axes.imshow(image, cmap=pyplot.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)
pyplot.tight_layout()
pyplot.show()

''' Split the data for training and testing'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=11)

# peek the sizes of each set 
print(X_train.shape)
print(X_test.shape)

''' Create the model '''
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

knn.fit(X=X_train, y=y_train)

predicted = knn.predict(X=X_test)
expected = y_test

### Verify and look for errors 
wrong = [(p, e) for (p, e) in zip(predicted, expected) if p != e]

''' Test the models accuracy '''
print(f'{knn.score(X_test, y_test):.2%}')

# Confusion matrix (shows hits and misses)
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_true=expected, y_pred=predicted)
print(confusion)

# Classification report 
from sklearn.metrics import classification_report
names = [str(digit) for digit in digits.target_names]
print(classification_report(expected, predicted, target_names=names))

# Extra visualization, Heat map!
import pandas as pd
import seaborn as sns

confusion_df = pd.DataFrame(confusion, index=range(10), columns=range(10))
axes = sns.heatmap(confusion_df, annot=True, cmap='nipy_spectral_r')

''' K-fold cross-valdation '''
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, random_state=11, shuffle=True)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=knn, X=digits.data, y=digits.target, cv=kfold)
print(scores)

print(f'Mean Accuracy: {scores.mean():.2%}')
print(f'Accuracy standard deviation: {scores.std():.2%}')

''' Testing multiple model methods to determine which is the best'''
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

estimators = {
    'KNeighborsClassifier': knn,
    'SVC': SVC(gamma='scale'),
    'GaussianNB': GaussianNB()
}

# create a for loop to handle them
for k in range(1, 20, 2):
    kfold = KFold(n_splits=10, random_state=11, shuffle=True)
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(estimator=knn, X=digits.data, y=digits.target, cv=kfold)
    print(f'k={k:<2}; mean accuracy = {scores.mean():.2%}; ' + f'standard deviation= {scores.std():.2%}')
    
