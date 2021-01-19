import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
import time
import matplotlib
import gtrend 
import pandas as pd
from pytrends.request import TrendReq
from pytrends import dailydata

plt.style.use('seaborn-darkgrid')
matplotlib.rcParams['font.family'] = ['Heiti TC']
def rmax(maxrow: int=50):
    pd.set_option('display.max_rows', maxrow)

def cmax(maxcol: int=50):
    pd.set_option('display.max_columns', maxcol)

keyword_set = ['bitcoin', 'ethereum']

for keyword in keyword_set:
    pytrend = TrendReq(hl='en-US')
    keyword = keyword
    start = '2017-01-01'
    end = '2021-01-15'
    geo='US'
    cat=0
    gprop=''

    overlapping = gtrend.get_daily_trend(pytrend, keyword, start, end, geo=geo, cat=cat, gprop=gprop, verbose=True, tz=0)

    start_d = datetime.strptime(start, '%Y-%m-%d')
    end_d = datetime.strptime(end, '%Y-%m-%d')
    s_year = start_d.year
    s_mon = start_d.month
    e_year = end_d.year
    e_mon = end_d.month

    dailydata = dailydata.get_daily_data(word= keyword,
                    start_year= s_year,
                    start_mon= s_mon,
                    stop_year= e_year,
                    stop_mon= e_mon,
                    geo= geo,
                    verbose= False,
                    wait_time = 1.0)

    dailydata[f'{keyword}'].plot(figsize=(30,20))
                    

# # Print current trending data 
# canada = pytrend.trending_searches(pn='canada')
# print(canada.head())

# united_states = pytrend.trending_searches(pn='united_states')
# print(united_states.head())



