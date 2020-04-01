# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:43:44 2020

@author: conno
"""


import yfinance as yf #conda install didn't work for me, pip install yfinance in the anaconda prompt worked
from datetime import datetime
from datetime import timedelta




def create_stock_df(companies, days_used, test_date):
    test_date=datetime.strptime(test_date,'%Y-%m-%d')
    historical_data = yf.download(companies, start=test_date-timedelta(days=days_used), end=test_date)


    #This loop gets our dataset to the correct amount of days if it is short due to weekends
    i=1
    while len(historical_data)<days_used:
        historical_data = yf.download(companies, start=test_date-timedelta(days=days_used+i), end=test_date)
        i+=1
    return historical_data

#companies=["MSFT", "AAPL"] #Enter all company abbreviations that you would like to monitor
#days_used = 30 #How many previous days do you want to use in creating your bounds
#test_date='2020-03-23' #Enter date in yyyy-mm-dd format
#
#
#stock_df = create_stock_df(companies,days_used,test_date)



