from scipy.sparse.construct import random
from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()

print(california.DESCR)

''' EDA '''
import pandas as pd 
pd.set_option('precision', 4)
pd.set_option('max_columns',9)
pd.set_option('display.width', None)

california_df = pd.DataFrame(california.data, columns= california.feature_names)

california_df['MedHouseValue']  = pd.Series(california.target)

print(california_df.head())
print(california_df.describe())

sample_df = california_df.sample(frac=0.1, random_state=17)

''' Visualize the data to get a better handle on it '''
import matplotlib.pyplot as pyplot
import seaborn as sns
sns.set(font_scale=2)
sns.set_style('whitegrid')

for feature in california.feature_names:
    pyplot.figure(figsize=(16,9))
    sns.scatterplot(data=sample_df, x=feature, y='MedHouseValue', hue='MedHouseValue', palette='cool', legend=False)

pyplot.show()

# This creates a graph for each feature and how the data is distributed

''' Split and train the model '''
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    california.data, california.target, random_state= 11
)

from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X=X_train, y=y_train)

for i, name in enumerate(california.feature_names):
    print(f'{name:>10} : {linear_regression.coef_[i]}')

''' Testing the model '''
predicted = linear_regression.predict(X_test)

expected = y_test

df = pd.DataFrame()

df['Expected'] = pd.Series(expected)
df['Predicted'] = pd.Series(predicted)

figure = pyplot.figure(figsize=(9,9))
axes = sns.scatterplot(data=df, x='Expected', y='Predicted', hue='Predicted', palette='cool', legend=False)

start = min(expected.min(), predicted.min())
end = max(expected.max(), predicted.max())

axes.set_xlim(start,end)
axes.set_ylim(start,end)

line = pyplot.plot([start, end], [start,end], 'k--')
pyplot.show()

''' Score the model (how well did it work? if at all...) '''
from sklearn import metrics
print(metrics.r2_score(expected, predicted))
print(metrics.mean_squared_error(expected, predicted))

''' Choose the best model; testing other models '''
from sklearn.linear_model import ElasticNet, Lasso, Ridge

estimators = {
    'Linear Regression': linear_regression,
    'ElasticNet': ElasticNet(),
    'Lasso': Lasso(),
    'Ridge': Ridge()
}

from sklearn.model_selection import KFold, cross_val_score

for estimator_name, estimator_object in estimators.items():
    kfold = KFold(n_splits=10, random_state=11, shuffle=True)
    scores = cross_val_score(estimator=estimator_object, X=california.data, y=california.target, cv=kfold, scoring='r2')
    print(f'{estimator_name:>16}: ' + f'mean of r2 scores = {scores.mean():.3f}')

