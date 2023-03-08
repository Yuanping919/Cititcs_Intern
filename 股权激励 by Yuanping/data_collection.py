# SIP: share incentive plan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
import os
import warnings
warnings.filterwarnings('ignore')

ts.set_token("d0099dbe40a16319c19a8a4d8cf9d9bb67aa0512f522e99289fd3495")
pro = ts.pro_api("d0099dbe40a16319c19a8a4d8cf9d9bb67aa0512f522e99289fd3495")

data = pd.read_excel('wind_sip.xlsx')
data = data[data['代码'].notna()]
data.sort_values('预案公告日', inplace=True)
stock_list = list(data['代码'].unique())
stock_s = ','.join(stock_list[0:1000])
for code in stock_list:
    if (code + '.csv') in os.listdir('fundamental_data'):
        continue
    temp_df = pro.query('daily', ts_code=code,start_date='20100101')
    temp_df.to_csv('stock_data/'+code + '.csv')
    temp_df = pro.query('daily_basic', ts_code=code,start_date='20100101')
    temp_df.to_csv('fundamental_data/'+code + '.csv')