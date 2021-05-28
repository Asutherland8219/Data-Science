import pandas as pd 
import pandas_datareader as dr

index_url = "https://en.wikipedia.org/wiki/Nasdaq-100#Components"
data_table = pd.read_html(index_url)

print(data_table[3])


tickers = data_table[3]['Ticker'].tolist()

print(tickers)

print("\n Total number of companies are:", len(tickers))



