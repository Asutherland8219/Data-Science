import pandas as pd 
import pandas_datareader as dr

index_url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
data_table = pd.read_html(index_url)


tickers = data_table[1]['Symbol'].tolist()

print(tickers)

print("\n Total number of companies are:", len(tickers))



