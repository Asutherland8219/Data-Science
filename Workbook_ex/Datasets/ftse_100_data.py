import pandas as pd 
import pandas_datareader as dr

index_url = "https://en.wikipedia.org/wiki/FTSE_100_Index#Constituents_in_May_2021"
data_table = pd.read_html(index_url)

tickers = data_table[3]['EPIC'].tolist()

print(tickers)

print("\n Total number of companies are:", len(tickers))

price_list = []
for ticker in tickers:
    try:
        prices = dr.DataReader(ticker, 'yahoo', '01/01/2018')['Close']
        prices = pd.DataFrame(prices)
        prices.columns = [ticker]
        price_list.append(prices)
    except:
        pass
    prices_df = pd.concat(price_list, axis=1)
prices_df.sort_index(inplace=True)

print(prices_df)

# prices_df.to_csv("ftse10data.csv")




