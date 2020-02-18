# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:01:16 2020

@author: conno
"""



import sys

import json
import urllib.request
from datetime import datetime
from time import sleep
from io import StringIO
from datetime import datetime
import pandas as pd
import boto3


token ='pk_871cfbde34a04823ba26850dfb9134b8'
bucket='bigolebucket12345'
tag = 'fb'

def _write_dataframe_to_csv_on_s3(dataframe, filename):
    """ Write a dataframe to a CSV on S3 """
    print("Writing {} records to {}".format(len(dataframe), filename))
    # Create buffer
    csv_buffer = StringIO()
    # Write dataframe to buffer
    dataframe.to_csv(csv_buffer, sep="|", index=False)
    # Create S3 object
    s3_resource = boto3.resource("s3")
    # Write buffer to S3 object
    s3_resource.Object(bucket, filename).put(Body=csv_buffer.getvalue())



n = 0
timestep=60
openseconds=(14-7.5)*60*60
records_to_be_recorded = openseconds/timestep


data=[]
while n<=records_to_be_recorded:
    
    try:
        html = urllib.request.urlopen("https://cloud.iexapis.com/stable/tops?token="+token+"&symbols="+tag)
        lst = json.loads(html.read().decode('utf-8'))
        symbol = lst[0]['symbol']
        BidPrice = lst[0]['bidPrice']
        BidSize = lst[0]['bidSize']
        AskPrice = lst[0]['askPrice']
        AskSize = lst[0]['askSize']
        lastUpdatedTime = lst[0]['lastUpdated']
        lastSalePrice = lst[0]['lastSalePrice']
        lastSaleSize = lst[0]['lastSaleSize']
        lastSaleTime = lst[0]['lastSaleTime']
        volume = lst[0]['volume']
        x=datetime.now()
        y=lastSalePrice
        

        n += 1
        data.append([symbol,lastSaleTime,lastSalePrice,volume])
        sleep(timestep)
    except:
        print('No return on iteration ',n)



todaytag=datetime.today().strftime('%Y-%m-%d')
df=pd.DataFrame(data,columns=['symbol','lastSaleTime','lastSalePrice','volume'])
_write_dataframe_to_csv_on_s3(df, tag+todaytag+'.csv')

     