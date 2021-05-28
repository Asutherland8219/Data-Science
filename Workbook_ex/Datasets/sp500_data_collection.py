import pandas as pd 
import pandas_datareader as dr

snp500url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
data_table = pd.read_html(snp500url)

tickers = data_table[0][1:]['Symbol'].tolist()

print(tickers)

print("\n Total number of companies are:", len(tickers))
