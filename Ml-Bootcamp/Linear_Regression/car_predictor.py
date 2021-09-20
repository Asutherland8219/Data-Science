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
