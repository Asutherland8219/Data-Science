from typing import OrderedDict
import pandas as pd 
import numpy as np 
from fastai.basics import *


df = pd.read_csv('Workbook_ex/Datasets/Kaggle_sets/blue-bulldozer/TrainAndValid.csv', low_memory=False)

print(df.columns)

# handle the unique columns 
print(df['ProductSize'].unique())

# order the list 
sizes = 'Large', 'Large / Medium ', 'Medium', 'Small', 'Mini', 'Compact'

df['ProductSize'] = df['ProductSize'].astype('category')
df['ProductSize'].cat.set_categories(sizes, ordered=True,inplace=True)


# isolate the most important variable, which in this case is the 'SalePrice'
dep_var = 'SalePrice'
df[dep_var] = np.log(df[dep_var])

df = add_datepart(df, 'saledate')

df_test = pd.read_csv('Data-Science/Workbook_ex/Datasets/Kaggle_sets/blue-bulldozer/Test.csv', low_memory=False)
df_test = add_datepart(df_test, 'saledate')

# join the new columns 
' '.join(o for o in df.columns if o.startswith('sale'))

procs = [Categorify, FillMissing]

cond = (df.saleYear<11)

#### incomplete

