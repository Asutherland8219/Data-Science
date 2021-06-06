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
        self.S_t_input = tf.placeholder(tf.float32, [time_steps, batch_size, features])
        self.K = tf.placeholder(tf.float32, batch_size)
        self.alpha = tf.placeholder(tf.float32)

        S_t = self.S_t_input[-1, :, 0]
        dS = self.S_t_input[1:, :, 0]

        # prepare S_t for the use in the RNN remove the last step (at T the portoflio is 0)
        S_t = tf.unstact(self.S_t_input[:-1, :, :], axis= 0 )

        # build the lstm
        lstm = tf.contrib.rnn.MultiRNNcell([tf.contrib.rnn.LSTMCell(n) for n in nodes ])

        self.strategy, state = tf.nn.static_rnn(lstm, S_t, initial_state=lstm.zero_state(batch_size, tf.float32), dtype=tf.float32)
        self.strategy = tf.reshape(self.strategy, (time_steps-1, batch_size))
        self.option = tf.maximum(S_T-self.K, 0)

        self.Hedging_PnL = - self.option + tf.reduce_sum(dS * self.strategy, axis = 0)
        self.Hedging_PnL_paths = -self.option + dS*self.strategy

        # Calculate cvar 
        CVaR, idx = tf.nn.top_k(-self.Hedging_PnL, tf.cast((1-self.alpha) * batch_size, tf.int32))
        CVaR = tf.reduce_mean(CVaR)
        self.train = tf.train.AdamOptimizer().minmize(CVaR)
        self.saver = tf.train.Saver()
        self.modelname = name 

    
    # this is the key function of the program 
    def _execute_graphy_batchwise(self, paths, strikes, riskaversion, sess, epochs=1, train_flag=False):
        sample_size = paths.shape[1]
        batch_size = self.batch_size
        idx = np.arrange(sample_size)
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
            sess.run(tf.global_variables_initializer())
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
with tf.Session() as sess:
    model_1.training(paths_train, np.ones(paths_train.shape[1]) * K, alpha, epoch, sess)
print('Training finished, Time elapsed:', dt.datetime.now()-start)

