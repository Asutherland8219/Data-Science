import numpy as py
import pandas as pd
import requests 
import xlsxwriter

from secrets import IEX_CLOUD_API_TOKEN


stocks = pd.read_csv('./Data-Science/Algo_IEX/sp_500_stocks.csv')

symbol = 'AAPL'
api_url = f'https://sandbox.iexapis.com/stable/stock/{symbol}/quote?token={IEX_CLOUD_API_TOKEN}'
data = requests.get(api_url).json()

price = data['latestPrice']
market_cap= data['marketCap']

my_columns = ['Ticker', 'Stock Price', 'Maret Capitalization', 'Number of Shares to Buy']
final_dataframe = pd.DataFrame([[0,0,0,0,]], columns = my_columns)
print(final_dataframe)



