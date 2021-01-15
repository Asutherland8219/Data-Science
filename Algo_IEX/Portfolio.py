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


