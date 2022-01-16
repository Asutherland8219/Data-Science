from site import setcopyright
import numpy as py
import pandas as pd
import requests 
import xlsxwriter
import json
from secrets import IEX_CLOUD_API_TOKEN, IEX_NEW_TOKEN

# create the empty list and begin the portfolio creation process


symbol = ['AAPL']

my_columns = ['Ticker', 'Company Name','Stock Price', 'Market Capitalization', 'Percent Change (Day)', 'PE Ratio', 'Volume', '52 High', '52 Low', 'Number of Shares to Buy']

df_ticker = pd.DataFrame(symbol)



final_dataframe = pd.DataFrame(columns = my_columns)
batch_api_call_url = f'https://cloud.iexapis.com/stable/stock/market/batch/?types=quote&symbols={symbol}&token={IEX_NEW_TOKEN}'
data = requests.get(batch_api_call_url).json()

# print(data)

# Get the important metrics to compare
# company_url = f'https://sandbox.iexapis.com/stable/stock/{symbol}/company/&token={IEX_CLOUD_API_TOKEN}'

# company_data = requests.get(company_url).json()

''' This is incluced in the payload in the following endpoint : GET /stock/{symbol}/company '''

''' issueType 	string 	Refers to the common issue type of the stock.
ad - ADR
cs - Common Stock
cef - Closed End Fund
et - ETF
oef - Open Ended Fund
ps - Preferred Stock
rt - Right
struct - Structured Product
ut - Unit
wi - When Issued
wt - Warrant
empty - Other'''

''' We are mainly concerned with the following: '''
# industry
# sector
# tags
# issueType

''' Once we have the data, we want to hit the collections endpoint and compare '''

''' 1st request make the ping for the base ticker 
    2nd request for list of Most active 
    3rd request for primary industry
    4th request for secondary industry 
    
    Finally compare 1st request to most active/primary/secondary and return top 5 '''

with open("Algo_IEX/json_loads/aapl.json", "r") as read_file:
    data = json.load(read_file)


primer = {}


primer['symbol'] = data.get('symbol')      
primer['exchange'] =data.get('exchange')
primer['industry'] =data.get('industry')
primer['issuetype'] = data.get('issueType')
primer['sector'] = data.get('sector')
total_tags = data.get('tags')

primer['primary'] = total_tags[0]
primer['secondary'] = total_tags[1]


''' Get a sector list to draw from '''
# api_call_sector = f"https://cloud.iexapis.com/stable/stock/market/ref-data/sectors?token={IEX_NEW_TOKEN}"
# sector_list = requests.get(api_call_sector)

# print(sector_list)

# with open('sector_data.json', 'w') as outfile:
#     json.dump(sector_list, outfile)

api_call_tech = f"https://cloud.iexapis.com/stable/stock/market/collection/sector?collectionName=Technology&token={IEX_NEW_TOKEN}"
tech_list = requests.get(api_call_tech).json()

# with open('tech_data.json', 'w') as outfile:
#     json.dump(tech_list, outfile)

api_most_active_tech = f"https://cloud.iexapis.com/stable/stock/market/list/mostactive?collectionName=mostactive&token={IEX_NEW_TOKEN}"
data_most = requests.get(api_most_active_tech).json()

# with open('most_active_data.json', 'w') as outfile:
#     json.dump(data_most, outfile)

api_cyclical= f"https://cloud.iexapis.com/stable/stock/market/collection/sector?collectionName=&token={IEX_NEW_TOKEN}"
data_list = requests.get(api_cyclical).json()

print(data_list)

with open('cyclical_data.json', 'w') as outfile:
    json.dump(data_list, outfile)











