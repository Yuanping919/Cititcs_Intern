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

data = pd.read_excel('data_05-20.xlsx')
data = data[data['代码'].notna()]
stock_list = list(data['代码'].unique())
stock_s = ','.join(stock_list[0:1000])
for code in stock_list:
    if (code + '.csv') in os.listdir('fundamental_data'):
        continue
    temp_df = pro.query('daily', ts_code=code,start_date='20100101')
    temp_df.to_csv('stock_data/'+code + '.csv')
    temp_df = pro.query('daily_basic', ts_code=code,start_date='20100101')
    temp_df.to_csv('fundamental_data/'+code + '.csv')


# data_plan = pd.read_excel("sip_data/plan/CG_EIPLdfAN.xlsx")
# det1 = pd.read_excel("sip_data/detail_part1/CG_EIVESTLIST.xlsx")
# det2 = pd.read_excel("sip_data/detail_part2/CG_EIAWARDLIST.xlsx")

##
"""
absolute return
"""
time_window = 120
abs_col = ['start_date', 'code'] + ['equity'+str(x+1) for x in range(time_window)] + ['index' + str(x+1) for x in range(time_window)]
abs_ret = pd.DataFrame(columns=abs_col)

"""
CSI：中证全指
"""
csi_data = pd.read_csv('000985.csv')
csi_data.trade_date = pd.to_datetime(csi_data.trade_date.astype(str))
csi_data.sort_values('trade_date', ascending=True, inplace=True)
for i in range(len(data)):
    temp_series = data.iloc[i,:]
    stock_data = pd.read_csv('stock_data/' + temp_series['代码'] + '.csv')
    stock_data.trade_date = pd.to_datetime(stock_data.trade_date.astype(str))
    stock_data.sort_values('trade_date', ascending=True, inplace=True)
    # temp_data['pre_close'] = temp_data['close'].shift(1)
    stock_data = stock_data[stock_data['trade_date'] >= temp_series['预案公告日']]
    stock_data['cost'] = stock_data['open'].iloc[0]
    stock_data['ret'] = stock_data['close'] / stock_data['cost'] - 1

    csi = csi_data.copy()
    csi = csi[csi.trade_date>=temp_series['预案公告日']]
    csi['cost'] = csi['open'].iloc[0]
    csi['ret'] = csi['close']/csi['cost']-1

    new_row = [temp_series['预案公告日'], temp_series['代码']] + list(stock_data.ret.iloc[0:time_window]) +\
              list(csi.ret.iloc[0:time_window])

    abs_ret.loc[i] = new_row



