import numpy as py
import pandas as pd 
import xlsxwriter 

btc = pd.read_csv('./Data-Science/BTC_Analysis/BTC-USD.csv')

my_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

dataframe = pd.DataFrame(columns= my_columns)

final_df = dataframe.append(btc, ignore_index = True)

final_df["Day_percent_change"] = ( final_df["Close"] - final_df["Open"] ) / final_df["Open"]

final_df["Day_percent_change_in_percent"] = (final_df["Day_percent_change"] * 100)

print(final_df)




