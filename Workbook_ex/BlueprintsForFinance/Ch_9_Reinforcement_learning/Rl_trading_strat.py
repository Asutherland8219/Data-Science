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

import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot

#Import Model Packages for reinforcement learning
from keras import layers, models, optimizers
from keras import backend as K
from collections import namedtuple, deque

# Import packages for creation of the agent 
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
from IPython.core.debugger import set_trace

import numpy as np
import random
from collections import deque




# diable the warnings
import warnings 
warnings.filterwarnings('ignore')

# load data set, check shape and stats
dataset = read_csv('Workbook_ex\Datasets\SP500.csv')

print(dataset.head())
print(dataset.shape)
set_option('precision', 3)
print(dataset.describe())

dataset['Close'].plot()

pyplot.show()

''' Data cleaning '''
# check for null values and replace them 
print('Null Values =', dataset.isnull().values.any())

# None in the set but here is function to fill 
dataset = dataset.fillna(method='ffill')

''' Evaluate Algos and models ''' 
# We split dataset in to modeling and testing at an 80/20 ratio respectively

X = list(dataset["Close"])
X = [float(x) for x in X]

validation_size = 0.2

train_size = int(len(X) * (1-validation_size))
X_train, X_test = X[0:train_size], X[train_size:len(X)]

''' Create classes and functions neeeded '''
## this model matches stats to actions 

class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        # State is dependent on the window size, n previous days 
        self.state_size = state_size
        self.action_size = 3 # hold, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.model = load_model(model_name) if is_eval else self._model()

    #Deep Q learning model 
    def _model(self):
        model = Sequential()
        # Input layer
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        # Hidden Layers 
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        # Output layer 
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return  model


    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state)
        return np.argmax(options[0])

    def expReplay(self,batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
        
        # memory training during phase
        for state, action, reward, next_state, done in mini_batch:
            target = reward

            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

        # initial Q
            target_f = self.model.predict(state)

        # update Q based on results or action 
            target_f[0][action] = target

        # train and fit the model where state is X and target_f is y
        self.model.fit(state, target_f, epochs=1, verbose= 0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

''' Create Helper functions '''
import numpy as np
import math

# price formatted
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the sigmoid 
def sigmoid(x):
    return 1/ (1 + math.exp(-x))

# returns as an n-day state representation ending at time t 
def getState(data, t, n): 
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]

    res = []
    for i in range (n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])

# plots behaviour of the output 
def plot_behaviour(data_input, states_buy, states_sell, profit):
    fig = pyplot.figure(figsize= (15, 5))
    pyplot.plot(data_input, color='r', lw=2.)
    pyplot.plot(data_input, '^', markersize=10, color='m', label= 'Buying signal', markevery = states_buy)
    pyplot.plot(data_input, 'v', markersize=10, color='k', label= 'Selling signal', markevery = states_sell)
    pyplot.title('Total gains : %f' % (profit))
    pyplot.legend()
    pyplot.show()

# create some graphs on the data 
from IPython.core.debugger import set_trace 
window_size = 1 
agent = Agent(window_size, False)

data = X_train
l = len(data) - 1

batch_size = 32
# a episode if a completepass over the data 
episode_count = 10 

for e in range(episode_count + 1):
    print("Running episode" + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)

    total_profit = 0
    agent.inventory = []
    states_sell = []
    states_buy = []
    for t in range(l):
        action = agent.act(state)

        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1: #buy
            agent.inventory.append(data[t])
            states_buy.append(t)
            print("Buy: " + formatPrice(data[t]))

        elif action == 2 and len(agent.inventory)  > 0: #sell
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            states_sell.append(t)
            print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

        done = True if t == l -1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("--------------------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------------------")

            plot_behaviour(data, states_buy, states_sell, total_profit)
        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    if e % 2 == 0:
        agent.model.save("model_ep" + str(e))

    ''' Test Data '''
    test_data = X_test
    l_test = len(test_data) - 1
    state = getState(test_data, 0, window_size + 1)
    total_profit = 0 
    is_eval = True
    done = False
    states_sell_test = []
    states_buy_test = []

    #get the trained model 
    model_name = "model_ep"+str(episode_count)
    agent = Agent(window_size, is_eval, model_name)
    state = getState(data, 0, window_size + 1)
    total_profit = 0 
    agent.inventory = []

for t in range(l_test):
    action = agent.act(state)

    next_state = getState(test_data, t + 1, window_size + 1)
    reward = 0

    if action == 1:
        agent.inventory.append(test_data[t])
        states_buy_test.append(t)
        print("Buy: " + formatPrice(test_data[t]))

    elif action == 2 and len(agent.inventory) > 0:
        bought_price = agent.inventory.pop(0)
        reward = max(test_data[t] - bought_price, 0)

        total_profit += test_data[t] - bought_price
        states_sell_test.append(t)
        print("Sell: " + formatPrice(test_data[t]) + " | profit:" + formatPrice(test_data[t] - bought_price))

    if t ==  l_test - 1 :
        done = True
    agent.memory.append((state, action, reward, next_state, done))
    state = next_state



        

    


        
        







