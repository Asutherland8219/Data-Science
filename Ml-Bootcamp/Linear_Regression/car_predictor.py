from sys import settrace
import pandas as pd 
import numpy as np 
import kaggle

# URL for download https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv

data = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv')

print(data.head())
df = data
print(df.columns)

df.columns = df.columns.str.lower().str.replace(' ', '_')
print(df.columns)

print(df.dtypes)

strings = list(df.dtypes[df.dtypes == 'object'].index)

for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_')

print(df.head())

''' We need to explore the columns further; we must go DEEPER '''
for col in df.columns:
    print(col)
    print(df[col].unique()[:10])
    print(df[col].nunique())
    print()


import matplotlib.pyplot as plt 
import seaborn as sns

sns.histplot(df.msrp, bins=50)
plt.show()

# some of these value are very large, so we need to reduce them using the LOG function (log1p)
price_logs = np.log1p(df.msrp)

sns.histplot(price_logs, bins=50)
plt.show()

''' Setting up the validation framework'''
# here wer are manually running the function "traintestplit"
n = len(df)

n_val = int(n*0.2)
n_test = int(n*0.2)
n_train = n - n_val - n_test


# shuffle the data so it is not sequential
# sometimes you might want to keep the same data in order for it to be reproduceable, for this you would use np.random.seed(2)
idx = np.arange(n)
np.random.shuffle(idx)

#np.random.seed(2)

# slice the DF based on the above numbers 
df_train = df.iloc[idx[:n_train]]   
df_val = df.iloc[idx[n_train:n_train + n_val]]
df_test = df.iloc[idx[n_train+n_val:]]


print(len(df_train), len(df_val), len(df_test))

# reset and drop index because we dont need it 
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_train = df_test.reset_index(drop=True)

print(df_train.head())

# separate the value you are trying to find 
y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)

# delete the columns to make it easier to predict the MSRP
del df_train['msrp']
del df_val['msrp']
del df_test['msrp']


print(df_train.head())


''' Train the linear regression model '''   
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]

''' Create a car baseline model '''
print(df_train.dtypes)
# seperate the numeric types 
base = ['engine_hp','engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']

# now we can return the subset of columns using base in order to verify 
print(df_train[base])

# always fill na values to not run into NAN issues 
# we can extract the values into a numpy array 
X_train = df_train[base].fillna(0).values

w0, w = train_linear_regression(X_train, y_train)

y_pred = w0 + X_train.dot(w)

sns.histplot(y_pred, color = 'red', alpha=0.5, bins=50)
sns.histplot(y_train, color = 'blue', alpha=0.5, bins=50)
plt.show()

# visually it looks like the model undervalues on prediction 

''' Create a RMSE to evaluate the model '''
def rmse(y , y_pred):
    error = y - y_pred
    se = error ** 2
    mse = se.mean()
    return np.sqrt(mse)

print(rmse(y_train, y_pred))

''' Feature engeineer a new feature based on the age of the cars '''
def prepare_X(df):
    df = df.copy()
    df['age'] = 2017 - df.year
    features = base + ['age']

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w) 

print(rmse(y_val, y_pred))
