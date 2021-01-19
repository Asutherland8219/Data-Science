from datetime import date, datetime, timedelta
from pytrends import dailydata

pytrend = TrendReq(hl='en-US')
keyword = 'bitcoin'
start = '2017-01-01'
end = '2020-01-15'
geo='US'
cat=0
gprop=''


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


