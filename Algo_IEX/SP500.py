import numpy as py
import pandas as pd
import requests 
import xlsxwriter

from secrets import IEX_CLOUD_API_TOKEN


stocks = pd.read_csv('sp_500_stocks.csv')

symbol = 'AAPL'
api_url = f'https://sandbox.iexapis.com/stable/stock/{symbol}/quote?token={IEX_CLOUD_API_TOKEN}'
data = requests.get(api_url).json()

ticker = data['symbol']
price = data['latestPrice']
market_cap= data['marketCap']
name = data['companyName']
market_cap = data['marketCap']
peratio = data['peRatio']
per_change = data['changePercent']
volume = data['volume']
price_high = data['week52High']
price_low = data['week52Low']
shares_buy = 100

stock_info = [{ticker, name, price, market_cap, per_change, peratio, volume, price_high, price_low, shares_buy}]



my_columns = ['Ticker', 'Company Name','Stock Price', 'Market Capitalization', 'Percent Change (Day)', 'PE Ratio', 'Volume', '52 High', '52 Low', 'Number of Shares to Buy']
final_dataframe = pd.DataFrame(stock_info, columns = my_columns)
print(final_dataframe)

