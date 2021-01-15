import numpy as py
import pandas as pd
import requests 
import xlsxwriter

from secrets import IEX_CLOUD_API_TOKEN


stocks = pd.read_csv('./Data-Science/Algo_IEX/sp_500_stocks.csv')

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


my_columns = ['Ticker', 'Company Name','Stock Price', 'Market Capitalization', 'Percent Change (Day)', 'PE Ratio', 'Volume', '52 High', '52 Low', 'Number of Shares to Buy']


final_dataframe = pd.DataFrame(columns = my_columns)


def chunks(lst, n):
    for i in range (0, len(lst), n):
        yield lst[i:i + n]

symbol_groups = list(chunks(stocks['Ticker'], 100))
symbol_strings = []

for i in range(0, len(symbol_groups)):
    symbol_strings.append(','.join(symbol_groups[i]))

for symbol_string in symbol_strings:
    batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch?symbols={symbol_string}&types=&token={IEX_CLOUD_API_TOKEN}'
data = requests.get(batch_api_call_url).json()
for symbol in symbol_string.split(','):

    final_dataframe = final_dataframe.append(
        pd.Series([symbol, 
        data[symbol]['quote']['latestPrice'], 
        data[symbol]['quote']['marketCap'], 
        'N/A'], 
        index = my_columns), 
        ignore_index = True)


final_dataframe = pd.DataFrame(columns = my_columns)






