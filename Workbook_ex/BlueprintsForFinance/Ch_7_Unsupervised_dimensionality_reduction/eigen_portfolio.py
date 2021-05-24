import numpy as np
from numpy.core.defchararray import index 
import pandas as pd 
import matplotlib.pyplot as pyplot
from pandas import read_csv, set_option
from pandas.io import pytables
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# model packages
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv, eig, svd

from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA

# load data set 
dataset= read_csv('Workbook_ex\Datasets\Dow_adjcloses.csv', index_col=0)

print(dataset.head())

print(dataset.shape)

correlation = dataset.corr()
pyplot.figure(figsize=(15, 15))
pyplot.title('Correlation Matrix')
sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
pyplot.show()


# Describe data
set_option('precision', 3)
print(dataset.describe())

''' Data Cleaning '''
missing_fractions = dataset.isnull().mean().sort_values(ascending=False)
print(missing_fractions.head(10))

drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
dataset.drop(labels=drop_list, axis=1, inplace=True)
print(dataset.shape)

# forward fill values 
dataset= dataset.fillna(method='ffill')

# could also drop na values 
# dataset= dataset.dropna()

print(dataset.head())

# compute daily return
datareturns= dataset.pct_change(1)
datareturns= datareturns[datareturns.apply(lambda x : (x-x.mean()).abs()<(3*x.std())).all(1)]

print(dataset.head())
print(datareturns.head())

## We need to scale the data set, all variables should be the same size for the dat set , thus needing to be scaled 
scaler = StandardScaler().fit(datareturns)
rescaledDataset = pd.DataFrame(scaler.fit_transform(datareturns), columns= datareturns.columns, index= datareturns.index)
# summarize the data
datareturns.dropna(how='any', inplace=True)
rescaledDataset.dropna(how='any', inplace=True)
print(rescaledDataset.head(2))

# visualize the data 
pyplot.figure(figsize=(16,5))
pyplot.title("AAPL return")
pyplot.ylabel("Return")
rescaledDataset.AAPL.plot()
pyplot.grid(True);
pyplot.legend()
pyplot.show()

''' Train test split data '''
percentage = int(len(rescaledDataset) * 0.8)
X_train = rescaledDataset[:percentage]
X_test = rescaledDataset[percentage:]

X_train_raw = datareturns[:percentage]
X_test_raw = datareturns[percentage:]

stock_tickers = rescaledDataset.columns.values
n_tickers = len(stock_tickers)

# Now we can add PCA 
pca = PCA()
PrincipalComponent = pca.fit(X_train)

print(pca.components_[0])

# Explained variance using PCA 
NumEigenvalues = 10 
fig, axes = pyplot.subplots(ncols=2, figsize=(14, 4))
Series1 = pd.Series(pca.explained_variance_ratio_[:NumEigenvalues]).sort_values()*100
Series2 = pd.Series(pca.explained_variance_ratio_[:NumEigenvalues]).cumsum()*100

Series1.plot.barh(ylim=(0,9), label= "woohoo", title='Explained Variance Ratio by Top 10 factors', ax=axes[0]);
Series2.plot(ylim=(0,100), xlim=(0,9), ax=axes[1], title='Cumulative Explained Variance by factor');

pd.Series(np.cumsum(pca.explained_variance_ratio_)).to_frame('Explained Variance').head(NumEigenvalues).style.format('{:,.2%}'.format)
pyplot.show()

# Looking at portfolio weights 
def PCWeights():

    weights = pd.DataFrame()
    for i in range(len(pca.components_)):
        weights["weights_{}".format(i)] = pca.components_[i] / sum(pca.components_[i])

    weights = weights.values.T
    return weights

weights = PCWeights()

print(sum(pca.components_[0]))

NumComponents = 5
topPortfolios = pd.DataFrame(pca.components_[:NumComponents], columns=dataset.columns)
eigen_portfolios = topPortfolios.div(topPortfolios.sum(1), axis=0)
eigen_portfolios.index = [f'Portfolio {i}' for i in range(NumComponents)]
np.sqrt(pca.explained_variance_)
eigen_portfolios.T.plot.bar(subplots=True, layout=(int(NumComponents), 1), figsize=(14,10), legend= False, sharey=True, ylim = (-1, 1))
pyplot.show()

sns.heatmap(topPortfolios)
pyplot.show()

''' Finding the best portfolio via sharpe ratio '''
# based off of convential trading days in a year (252)
def sharpe_ratio(ts_returns, periods_per_year=252):
    n_years = ts_returns.shape[0]/periods_per_year
    annualized_return = np.power(np.prod(1+ts_returns), (1/n_years))-1
    annualized_vol = ts_returns.std() * np.sqrt(periods_per_year)
    annualized_sharpe = annualized_return / annualized_vol
    return annualized_return, annualized_vol, annualized_sharpe

def optimizedPortfolio():
    n_portfolios = len(pca.components_)
    annualized_ret = np.array([0.] * n_portfolios)
    sharpe_metric = np.array([0.] * n_portfolios)
    annualized_vol = np.array([0.] * n_portfolios)
    highest_sharpe = 0
    stock_tickers = rescaledDataset.columns.values
    n_tickers = len(stock_tickers)
    pcs= pca.components_

    for i in range(n_portfolios):

        pc_w = pcs[i] / sum(pcs[i])
        eigen_prtfi = pd.DataFrame(data={'weights': pc_w.squeeze()*100}, index = stock_tickers)
        eigen_prtfi.sort_values(by=['weights'], ascending=False, inplace=True)
        eigen_prtfi_returns = np.dot(X_train_raw.loc[:, eigen_prtfi.index], pc_w)
        eigen_prtfi_returns = pd.Series(eigen_prtfi_returns.squeeze(), index=X_train_raw.index)
        er, vol, sharpe = sharpe_ratio(eigen_prtfi_returns)
        annualized_ret[i] = er
        annualized_vol[i] = vol
        sharpe_metric[i] = sharpe

        sharpe_metric = np.nan_to_num(sharpe_metric)

    # find highest sharpe
    highest_sharpe = np.argmax(sharpe_metric)

    print('Eigen portfolio #%d with the highest Sharpe. Return %.2f%%, vol= %.2f%%, Sharpe= %.2f' % (highest_sharpe, annualized_ret[highest_sharpe]*100, annualized_vol[highest_sharpe]*100, sharpe_metric[highest_sharpe]))

    fig, ax = pyplot.subplots()
    fig.set_size_inches(12, 4)
    ax.plot(sharpe_metric, linewidth=3)
    ax.set_title('Sharpe ratio of eigen-portfolios')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_xlabel('Portfolios')

    results = pd.DataFrame(data={'Return': annualized_ret,
    'Vol': annualized_vol,
    'Sharpe': sharpe_metric})
    results.dropna(inplace=True)
    results.sort_values(by=['Sharpe'], ascending=False, inplace=True)
    print(results.head(5))

    pyplot.show()

print(optimizedPortfolio())

# portfolio the best model, but what is the allocation to each assset?
weights= PCWeights()
portfolio = portfolio = pd.DataFrame()

def plotEigen(weights, plot=False, portfolio=portfolio):
    portfolio = pd.DataFrame(data= {'weights': weights.squeeze()*100}, index= stock_tickers)
    portfolio.sort_values(by=['weights'], ascending= False, inplace= True)
    if plot:
        portfolio.plot(title='Current Eigen-Portfolio Weights', figsize= (12, 6), xticks= range(0, len(stock_tickers), 1), rot=45, linewidth=3)
        pyplot.show()

    return  portfolio

plotEigen(weights=weights[0], plot=True)

''' Back testing the portfolios '''

# we will now try and backtest the portfolios in order to see how much better it would have performed under the eigen method vs equal weighted 

def Backtest(eigen):
    eigen_prtfi = pd.DataFrame(data= {'weights' : eigen.squeeze()}, index= stock_tickers)
    eigen_prtfi.sort_values(by=['weights'], ascending=False, inplace=True)
    eigen_prtfi_returns = np.dot(X_test_raw.loc[:, eigen_prtfi.index], eigen)
    eigen_portfolio_returns= pd.Series(eigen_prtfi_returns.squeeze(), index=X_test_raw.index)
    returns, vol, sharpe = sharpe_ratio(eigen_portfolio_returns)
    print('Current Eigen-Portfolio: \nReturn = %.2f%%\nVolatility = %2f.%%\n\Sharpe = %.2f' % (returns * 100, vol * 100, sharpe))
    equal_weight_return=(X_test_raw * (1/len(pca.components_))).sum(axis=1)
    df_plot = pd.DataFrame({'EigenPortfolio Return': eigen_portfolio_returns, 'Equal Weight Index': equal_weight_return}, index=X_test.index)
    np.cumprod(df_plot + 1).plot(title='Returns of the Equal Weighted index vs First eigen portfolio', figsize=(12, 6), linewidth=3)

    pyplot.show()

Backtest(eigen=weights[5])
Backtest(eigen=weights[1])
Backtest(eigen=weights[14])
Backtest(eigen=weights[20])

                                                        












