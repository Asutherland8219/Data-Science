''' Using the NYC data set regarding the weather, we can predict the weather for the next day '''
import pandas as  pd 
nyc = pd.read_csv('Workbook_ex/IntrotoPython/Ch_15_IntroML/ave_hi_nyc_jan_1895-2018.csv')
nyc.columns = ['Date', 'Temperature', 'Anomoly']
nyc.Date = nyc.Date.floordiv(100)

print(nyc.head())
 
''' Split the data set '''
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    nyc.Date.values.reshape(-1,1), nyc.Temperature.values, random_state=11
)

# check the split 
print(X_train.shape)
print(X_test.shape)

''' Train the model '''
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X=X_train, y=y_train)

predicted = linear_regression.predict(X_test)

expected = y_test

for p, e in zip(predicted[::5], expected[::5]):
    print(f'predicted: {p:.2f}, expected: {e:.2f}, differential: {(((p-e)/p)*100):.2f}')

''' Prediction equation '''
predict = (lambda x: linear_regression.coef_* x + linear_regression.intercept_)

''' Visualize the dataset '''
import seaborn as sns
axes = sns.scatterplot(data=nyc, x='Date', y='Temperature', hue='Temperature', palette='winter', legend=False)
axes.set_ylim(10, 70)

#visualize regression line 
import numpy as np 
x = np.array([min(nyc.Date.values), max(nyc.Date.values)])
y = predict(x)

import matplotlib.pyplot as pyplot
line = pyplot.plot(x,y)
pyplot.show()

