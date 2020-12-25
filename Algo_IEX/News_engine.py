import numpy as py
import pandas as pd
import requests 
import xlsxwriter

from secrets import IEX_CLOUD_API_TOKEN


stocks = pd.read_csv('./Data-Science/Algo_IEX/sp_500_stocks.csv')

symbol = 'AAPL'
api_url = f'https://cloud.iexapis.com/stable/time-series/news/{symbol}?token={IEX_CLOUD_API_TOKEN}/last'
data = requests.get(api_url)

print(data)

