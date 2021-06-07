# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import datetime as dt
import random
import scipy.stats as stats
import seaborn as sns
from IPython.core.debugger import set_trace

#Import Model Packages for reinforcement learning
from keras import layers, models, optimizers
from keras import backend as K
import tensorflow as tf
from tensorflow.python.framework.composite_tensor import replace_composites_with_components
tf.compat.v1.disable_eager_execution()
from tensorflow.python.framework import ops
from collections import namedtuple, deque

# disable the warnings
import warnings

from tensorflow.python.ops.gen_batch_ops import batch
from tensorflow.python.ops.gen_linalg_ops import BatchSelfAdjointEig 
warnings.filterwarnings('ignore')

''' Generating the Data ''' 
# her we are creating the monte carlo simulation paths 

def monte_carlo_paths(S_0, time_to_expiry, sigma, drift, seed, n_sims, n_timestamps):
    ''' Create random pahts following a browian geometic motion '''
    if seed > 0:
        np.random.seed(seed)
    stdnorm_random_variates = np.random.randn(n_sims, n_timestamps)
    S = S_0
    dt = time_to_expiry / stdnorm_random_variates.shape[1]
    r = drift 
    S_T = S * np.cumprod(np.exp((r-sigma**2/2)* dt+sigma*np.sqrt(dt)*stdnorm_random_variates), axis= 1)
    return np.reshape(np.transpose(np.c_[np.ones(n_sims)*S_0, S_T]), (n_timestamps+1,n_sims, 1))

S_0 = 100
K = 100
r = 0
vol = 0.2
T = 1/12
timesteps = 30
seed = 42
n_sims=50000

paths_train= monte_carlo_paths(S_0, T, vol, r, seed, n_sims, timesteps)

''' EDA ''' 

pyplot.figure(figsize=(20, 10))
pyplot.plot(paths_train[1:31,1])
pyplot.xlabel('Time Steps')
pyplot.title('Stock Price Sample Paths')
pyplot.show()

from IPython.core.debugger import set_trace 

class Agent(object):
    def __init__(self, time_steps, batch_size, features, nodes= [64, 46, 46, 1], name= 'model'):
        ops.reset_default_graph()
        self.batch_size = batch_size 
        self.S_t_input = tf.compat.v1.placeholder(tf.float32, [time_steps, batch_size, features])
        self.K = tf.compat.v1.placeholder(tf.float32, batch_size)
        self.alpha = tf.compat.v1.placeholder(tf.float32)

        S_T = self.S_t_input[-1, :, 0]
        dS = self.S_t_input[1:, :, 0] - self.S_t_input[0:-1, :, 0]

        # prepare S_t for the use in the RNN remove the last step (at T the portoflio is 0)
        S_t = tf.unstack(self.S_t_input[:-1, :, :], axis= 0 )

        # build the lstm
        lstm = tf.compat.v1.nn.rnn_cell.MultiRNNCell([tf.compat.v1.nn.rnn_cell.LSTMCell(n) for n in nodes ])

        self.strategy, state = tf.compat.v1.nn.static_rnn(lstm, S_t, initial_state=lstm.zero_state(batch_size, tf.float32), dtype=tf.float32)
        self.strategy = tf.reshape(self.strategy, (time_steps-1, batch_size))
        self.option = tf.maximum(S_T-self.K, 0)

        self.Hedging_PnL = - self.option + tf.reduce_sum(input_tensor=dS * self.strategy, axis = 0)
        self.Hedging_PnL_paths = -self.option + dS*self.strategy

        # Calculate cvar 
        CVaR, idx = tf.nn.top_k(-self.Hedging_PnL, tf.cast((1-self.alpha) * batch_size, tf.int32))
        CVaR = tf.reduce_mean(input_tensor=CVaR)
        self.train = tf.compat.v1.train.AdamOptimizer().minimize(CVaR)
        self.saver = tf.compat.v1.train.Saver()
        self.modelname = name 

    
    # this is the key function of the program 
    def _execute_graphy_batchwise(self, paths, strikes, riskaversion, sess, epochs=1, train_flag=False):
        sample_size = paths.shape[1]
        batch_size = self.batch_size
        idx = np.arange(sample_size)
        start = dt.datetime.now()
        for epoch in range(epochs):
            pnls = []
            strategies = []
            if train_flag:
                np.random.shuffle(idx)
            for i in range(int(sample_size/batch_size)):
                indices = idx[ i * batch_size : (i + 1) * batch_size]
                batch = paths[:, indices, :]
                if train_flag:
                    _, pnl, strategy = sess.run([self.train, self.Hedging_PnL, self.strategy], {self.S_t_input: batch, self.K : strikes[indices], self.alpha: riskaversion})
                
                else: pnl, strategy = sess.run([self.Hedging_PnL, self.strategy], {self.S_t_input: batch, self.K : strikes[indices], self.alpha: riskaversion})
                pnls. append(pnl)
                strategies.append(strategy)


            #Calculate option price given alpha
            CVaR = np.mean(-np.sort(np.concatenate(pnls)) [:int((1 - riskaversion) * sample_size)])

            if train_flag:
                if epoch % 10 == 0:
                    print('Time Elapsed:', dt.datetime.now()-start)
                    print('Epoch', epoch, 'CVaR', CVaR)
                    self.saver.save(sess, 'model.ckpt')
        self.saver.save(sess,'model.ckpt')
        return CVaR, np.concatenate(pnls), np.concatenate(strategies, axis=1)


    def training(self, paths, strikes, riskaversion, epochs, session, init=True):
        if init:
            sess.run(tf.compat.v1.global_variables_initializer())
        self._execute_graphy_batchwise(paths, strikes, riskaversion, session, epochs, train_flag=True)

    def predict(self, paths, strikes, riskaversion, session):
        return self._execute_graphy_batchwise(paths, strikes, riskaversion, session, 1, train_flag=False)
    
    def restore(self, session, checkpoint):
        self.saver.restore(session, checkpoint)

''' Train the Data  ''' 
batch_size = 1000
features = 1
K = 100
alpha = 0.50
epoch = 100
model_1 = Agent(paths_train.shape[0], batch_size, features, name='rnn_final')

start =  dt.datetime.now()
with tf.compat.v1.Session() as sess:
    model_1.training(paths_train, np.ones(paths_train.shape[1]) * K, alpha, epoch, sess)
print('Training finished, Time elapsed:', dt.datetime.now()-start)

''' Testing the Data ''' 
# Create black scholes model and function 
def BS_d1(S, dt, r, sigma, K):
    return (np.log(S/K) + (r+sigma**2/2)*dt) / (sigma*np.sqrt(dt))

def BlackScholes_price(S, T, r, sigma, K, t=0):
    dt = T-t
    Phi = stats.norm(loc = 0, scale=1).cdf
    d1 = BS_d1(S, dt, r, sigma, K)
    d2 = d1 - sigma * np.squrt(dt)
    return s*Phi(d1) - K * np.exp(-r * dt) * Phi(d2)

def black_scholes_hedge_strategy(S_0, K, r, vol, T, paths, alpha, output):
    bs_price = BlackScholes_price(S_0, T, r, vol, K, 0)
    times = np.zeros(paths.shape[0])
    times[1:] = T / (paths.shape[0]-1)
    times = np.cumsum(times)    
    bs_deltas = np.zeros((paths.shape[0]-1, paths.shape[1]))
    for i in range(paths.shape[0]-1):
        t = times[i]
        bs_deltas[i,:] = BS_delta(paths[i,:,0], T, r, vol, K, t)
    return test_hedging_strategy(bs_deltas, paths, K, bs_price, alpha, output)

def BS_delta(S, T, r, sigma, K, t=0):
    dt = T-t
    d1 = BS_d1(S, dt, r, sigma, K)
    Phi = stats.norm(loc=0, scale= 1).cdf
    return Phi(d1)

def test_hedging_strategy(deltas, paths, K, price, alpha, output=True):
    S_returns = paths[1:, :, 0]-paths[:-1, :, 0]
    hedge_pnl = np.sum(deltas * S_returns, axis=0)
    option_payoff = np.maximum(paths[-1, :, 0] - K, 0)
    replication_portfolio_pnls = -option_payoff + hedge_pnl + price
    mean_pnl = np.mean(replication_portfolio_pnls)
    cvar_pnl = -np.mean(np.sort(replication_portfolio_pnls)[:int((1-alpha) * replication_portfolio_pnls.shape[0])])
    if output:
        pyplot.hist(replication_portfolio_pnls)
        print('BS price at t0:', price)
        print('Mean Hedging PnL:', mean_pnl)
        print('CVaR Hedging PnL:',cvar_pnl)
    return(mean_pnl, cvar_pnl, hedge_pnl, replication_portfolio_pnls, deltas)

def plot_deltas(paths, deltas_bs, deltas_rnn, times = [0, 1,5,10,15,29]):
    fig = pyplot.figure(figsize=(10, 6))
    for i, t in enumerate(times):
        pyplot.subplot(2, 3, i+1)
        xs = paths[t, :, 0]
        ys_bs = deltas_bs[t, :]
        ys_rnn = deltas_rnn[t,:]
        df = pd.DataFrame([xs, ys_bs, ys_rnn]).T
        #df = df.groupby(0, as_index=False).agg({1:np.mean,
        #                                          2: np.mean})
        pyplot.plot(df[0], df[1], df[0], df[2], linestyle='', marker='x' )
        pyplot.legend(['BS delta', 'RNN Delta'])
        pyplot.title('Delta at Time %i' % t)
        pyplot.xlabel('Spot')
        pyplot.ylabel('$\Delta$')
    pyplot.tight_layout()
    
def plot_strategy_pnl(portfolio_pnl_bs, portfolio_pnl_rnn):
    fig = pyplot.figure(figsize=(10,6))
    sns.boxplot(x=['Black-Scholes', 'RNN-LSTM-v1 '], y=[portfolio_pnl_bs, portfolio_pnl_rnn])
    pyplot.title('Compare PnL Replication Strategy')
    pyplot.ylabel('PnL')
        

# test at 99% CVar
S_0 = 100
K = 100
r = 0 
vol = 0.2
T = 1/12
timesteps = 30 
seed_test = 21122017
n_sims_test = 10000

alpha = 0.99
paths_test = monte_carlo_paths(S_0, T, vol, r , seed_test, n_sims_test, timesteps)

with tf.Session() as sess:
    model_1.restore(sess, 'model.ckpt')
    test1_results = model_1.predict(paths_test, np.ones(paths_test.shape[1]) * K, alpha, sess)

_,_,_,portfolio_pnl_bs, deltas_bs = black_scholes_hedge_strategy(S_0, K, r, vol, T, paths_test, alpha, True)
pyplot.figure()
_,_,_,portfolio_pnl_rnn,deltas_rnn = test_hedging_strategy(test1_results[2], paths_test, K, 2.302974467802428, alpha, True)
plot_deltas(paths_test, deltas_bs, deltas_rnn)
plot_strategy_pnl(portfolio_pnl_bs, portfolio_pnl_rnn)


# changing Moneyness 
with tf.Session() as sess:
    model_1.restore(sess, 'model.ckpt')
    test1_results = model_1.predict(paths_test, np.ones(paths_test.shape[1]) * (K-10), alpha, sess)

_,_,_,portfolio_pnl_bs, deltas_bs = black_scholes_hedge_strategy(S_0, K-10, r, vol, T, paths_test, alpha, True)
pyplot.figure()
_,_,_,portfolio_pnl_rnn,deltas_rnn = test_hedging_strategy(test1_results[2], paths_test, K-10, 2.302974467802428, alpha, True)
plot_deltas(paths_test, deltas_bs, deltas_rnn)
plot_strategy_pnl(portfolio_pnl_bs, portfolio_pnl_rnn)

# changing Drift 
paths_test_drift = monte_carlo_paths(S_0, T, vol, 0.48 + r, seed_test, n_sims_test, timesteps)
with tf.Session() as sess:
    model_1.restore(sess, 'model.ckpt')
    test1_results = model_1.predict(paths_test_drift, np.ones(paths_test_drift.shape[1]) * K, alpha, sess)

_,_,_,portfolio_pnl_bs, deltas_bs = black_scholes_hedge_strategy(S_0, K, r, vol, T, paths_test_drift, alpha, True)
pyplot.figure()
_,_,_,portfolio_pnl_rnn,deltas_rnn = test_hedging_strategy(test1_results[2], paths_test_drift, K, 2.302974467802428, alpha, True)
plot_deltas(paths_test, deltas_bs, deltas_rnn)
plot_strategy_pnl(portfolio_pnl_bs, portfolio_pnl_rnn)

# Shifted/changed volatitlity 
paths_test_vol = monte_carlo_paths(S_0, T, vol + 0.05, r, seed_test, n_sims_test, timesteps)

with tf.Session() as sess:
    model_1.restore(sess, 'model.ckpt')
    test1_results = model_1.predict(paths_test_vol, np.ones(paths_test_vol.shape[1]) * K, alpha, sess)

_,_,_,portfolio_pnl_bs, deltas_bs = black_scholes_hedge_strategy(S_0, K, r, vol, T, paths_test_vol, alpha, True)
pyplot.figure()
_,_,_,portfolio_pnl_rnn,deltas_rnn = test_hedging_strategy(test1_results[2], paths_test_vol, K, 2.302974467802428, alpha, True)
plot_deltas(paths_test_vol, deltas_bs, deltas_rnn)
plot_strategy_pnl(portfolio_pnl_bs, portfolio_pnl_rnn)











