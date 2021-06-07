# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import datetime
import math
from numpy.random import choice
import random

from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model
from keras.regularizers import l2

import numpy as np
import pandas as pd

import random
from collections import deque
import matplotlib.pylab as pyplot

# Disable the warnings 
import warnings 
warnings.filterwarnings('ignore')

# load the data 
dataset = read_csv('Workbook_ex\Datasets\crypto_portfolio.csv')

print(dataset.shape)

set_option('display.width', 100)
print(dataset.head())

