import numpy as py
import pandas as pd
import requests 
import xlsxwriter

from secrets import IEX_CLOUD_API_TOKEN


stocks = pd.read_csv('./Data-Science/Algo_IEX/sp_500_stocks.csv')




my_columns = ['Ticker', 'Stock Price', 'Market Capitalization', 'Number of Shares to Buy']



def chunks(lst, n):
    for i in range (0, len(lst), n):
        yield lst[i:i + n]

symbol_groups = list(chunks(stocks['Ticker'], 100))
symbol_strings = []

print(symbol_groups)

for i in range(0, len(symbol_groups)):
    symbol_strings.append(','.join(symbol_groups[i]))

final_dataframe = pd.DataFrame(columns = my_columns)

for symbol_string in symbol_strings:
    batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch/?types=quote&symbols={symbol_string}&token={IEX_CLOUD_API_TOKEN}'
data = requests.get(batch_api_call_url).json()
for symbol in symbol_string.split(','):

    final_dataframe = final_dataframe.append(
        pd.Series([symbol, 
        data[symbol]['quote']['latestPrice'], 
        data[symbol]['quote']['marketCap'], 
        'N/A'], 
        index = my_columns), 
        ignore_index = True)


print(final_dataframe.head())






