import numpy as py
import pandas as pd
import requests 
import xlsxwriter

from secrets import IEX_CLOUD_API_TOKEN

# create the empty list and begin the portfolio creation process

input_string = input('Enter the assets you would like to purchase seperated by a comma ')
ticker_list = input_string.split(",")

my_columns = ['Ticker', 'Company Name','Stock Price', 'Market Capitalization', 'Percent Change (Day)', 'PE Ratio', 'Volume', '52 High', '52 Low', 'Number of Shares to Buy']

ticker_df = pd.DataFrame(ticker_list)

for symbol in ticker_df:
    api_url = f'https://sandbox.iexapis.com/stable/stock/{symbol}/quote?token={IEX_CLOUD_API_TOKEN}'
    data = requests.get(api_url)
  

    print(data)
    final_dataframe = ticker_df.append(
        pd.Series(
            [
                data['symbol'],
                data['companyName'],
                data['latestPrice'],
                data['marketCap'],
                data['changePercent'],
                data['peRatio'],
                data['volume'],
                data['week52High'],
                data['week52Low'],
            'N/A'],
        index= my_columns),
    ignore_index=True
    )

#/


# symbol = 'AAPL'
# api_url = f'https://sandbox.iexapis.com/stable/stock/{symbol}/quote?token={IEX_CLOUD_API_TOKEN}'
# data = requests.get(api_url).json()

# ticker = data['symbol']
# price = data['latestPrice']
# market_cap= data['marketCap']
# name = data['companyName']
# market_cap = data['marketCap']
# peratio = data['peRatio']
# per_change = data['changePercent']
# volume = data['volume']
# price_high = data['week52High']
# price_low = data['week52Low']
# shares_buy = 100


# my_columns = ['Ticker', 'Company Name','Stock Price', 'Market Capitalization', 'Percent Change (Day)', 'PE Ratio', 'Volume', '52 High', '52 Low', 'Number of Shares to Buy']





# def chunks(lst, n):
#     for i in range (0, len(lst), n):
#         yield lst[i:i + n]

# symbol_groups = list(chunks(stocks['Ticker'], 100))
# symbol_strings = []

# for i in range(0, len(symbol_groups)):
#     symbol_strings.append(','.join(symbol_groups[i]))

# final_dataframe = pd.DataFrame(columns = my_columns)

# for symbol_string in symbol_strings:
#     batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch/?types=quote&symbols={symbol_string}&token={IEX_CLOUD_API_TOKEN}'
# data = requests.get(batch_api_call_url).json()
# for symbol in symbol_string.split(','):

#     final_dataframe = final_dataframe.append(
#         pd.Series([symbol, 
#         data[symbol]['quote']['latestPrice'], 
#         data[symbol]['quote']['marketCap'], 
#         'N/A'], 
#         index = my_columns), 
#         ignore_index = True)


# final_dataframe = pd.DataFrame(columns = my_columns)

# print(final_dataframe.head())




