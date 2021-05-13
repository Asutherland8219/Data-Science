## Dataset can be found at https://oreil.ly/CeFRs
## in this model, class or the classification model is binary, 1 if fraudulant and 0 if otherwise 

''' Determining whether a transaction is fraudulent or not  '''

''' Function and modules for the Supervised learning Classification models '''
from re import VERBOSE
from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
from sklearn.model_selection import train_test_split, KFold, cross_val_score , GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.ops.array_ops import _cast_nested_seqs_to_dtype

''' Function and modules for Data Analysis and Model Evaluation '''
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression

''' Function and modules for deep learning models '''
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasRegressor

'''Function and modules for  time series models'''
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm 

'''Function and modules for data preparation and visualization'''

# standard data science imports
import numpy as np
import pandas as pd 
import pandas_datareader.data as web
import matplotlib as plt
import copy
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from pandas import read_csv, set_option
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
from statsmodels.graphics.tsaplots import plot_acf


# import to save the model
from pickle import dump
from pickle import load 

dataset = pd.read_csv('Data-Science\Workbook_ex\BlueprintsForFinance\Ch_6_Supervised_learning_Classification\creditcard.csv')
print(dataset.shape)

set_option('display.width', 100)
print(dataset.head(5))

class_names = {0:'Not Fraud', 1:'Fraud'}
print(dataset.Class.value_counts().rename(index= class_names))

## very few fraud cases to non fraud, too small to use, need to adjust data later

''' No Data Vis for this study as little insight is to be had; due to  its origin on Kaggle the data is also pre cleaned'''

''' Evaluate Models '''
Y = dataset["Class"]
X = dataset.loc[:, dataset.columns != 'Class']
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y, test_size=validation_size, random_state=seed)

## Use ten fold cross validation to evaluate models
num_folds = 10
scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# comopare the models
fig = pyplot.figure()
fig.suptitle ('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
fig.set_size_inches(8,4)
pyplot.show()

## The book uses CART but LDA (Linear) has the highest accuracy so im going to use that
# prepare the model
model = LinearDiscriminantAnalysis()
model.fit(X_train, Y_train)

#estimate the accuracy 
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

## Create a confusion matrix
df_cm = pd.DataFrame(confusion_matrix(Y_validation, predictions), columns=np.unique(Y_validation), index = np.unique(Y_validation))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size":16})
pyplot.show()

### adding CART calculations because Non linear
model1 = DecisionTreeClassifier()
model1.fit(X_train, Y_train)

predictions1 = model1.predict(X_validation)
print(accuracy_score(Y_validation, predictions1))
print(classification_report(Y_validation, predictions1))

## Create a confusion matrix
df_cm1 = pd.DataFrame(confusion_matrix(Y_validation, predictions1), columns=np.unique(Y_validation), index = np.unique(Y_validation))
df_cm1.index.name = 'Actual'
df_cm1.columns.name = 'Predicted'
sns.heatmap(df_cm1, cmap="Blues", annot=True, annot_kws={"size":16})
pyplot.show()

''' Model Tuning '''
### the most important feature is recall, so we will cross validate recall score 

scoring = 'recall'

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

### Sometimes we need to undersample a model, that is we remove data 
df = pd.concat([X_train, Y_train], axis=1)
# amount of fraud classes is 492 rows
fraud_df = df.loc[df['Class']== 1]
non_fraud_df = df.loc[df['Class']== 0][:492]

normal_distrubuted_df = pd.concat([fraud_df, non_fraud_df])

#shuffle rows of dataframe
df_new = normal_distrubuted_df.sample(frac=1, random_state=42)
##split out validation set for the end
Y_train_new = df_new["Class"]
X_train_new = df_new.loc[:, dataset.columns != 'Class']

#peek the new dataset distribution
print('Distribution of Classes in the subsample dataset')
print(df_new['Class'].value_counts()/len(df_new))
sns.countplot(x='Class', data=df_new)
pyplot.title('Equally Distriubuted Classes', fontsize=14)
pyplot.show()


# Now the data is closer balanced (55/44), close enough i think at least, it is now time to train the model 

#set the evaluation metric
scoring= 'accuracy'
#spot check the algos
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Neural Network
models.append(('NN', MLPClassifier()))

# Ensemble models 
models.append(('AB', AdaBoostClassifier()))

# Boosting 
models.append(('AB', AdaBoostClassifier()))
models.append(('GBM', GradientBoostingClassifier()))

# Bagging Methods
models.append(('RF', RandomForestClassifier()))
models.append(('ET', ExtraTreesClassifier()))

## Create Model function via Keras
EnableDLModelsFlag = 1
if EnableDLModelsFlag == 1 :
    def create_model(neurons=12, activation='relu', learn_rate=0.01, momentum=0):
        # create the model
        model = Sequential()
        model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1],activation=activation))
        model.add(Dense(32, activation=activation))
        model.add(Dense(1, activation='sigmoid'))
        # compile the model
        optimizer = SGD(lr= learn_rate, momentum=momentum)
        model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['accuracy'])
        return model
models.append(('DNN', KerasClassifier(build_fn=create_model, epochs=50, batch_size=10, verbose=0)))

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train_new, Y_train_new, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f) " % (name, cv_results.mean(), cv_results.std())
    print(msg)

# plot and comapre algos
fig = pyplot.figure()
fig.suptitle('Algo Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
fig.set_size_inches(8,4)
pyplot.show()

# boost the models

n_estimators = [20, 180, 1000]
max_depth = [2, 3, 5]
param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)
model = GradientBoostingClassifier()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X_train_new, Y_train_new)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

''' Prep the final model and check test results set '''

# prepare model by using the numbers provided in the last results
model = GradientBoostingClassifier(max_depth=5, n_estimators=1000)
model.fit(X_train_new, Y_train_new)
# estimate vs original set 
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

df_cm = pd.DataFrame(confusion_matrix(Y_validation, predictions), columns=np.unique(Y_validation), index = np.unique(Y_validation))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
pyplot.show()




