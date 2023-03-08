import pandas as pd
import tushare as ts
ts.set_token("d0099dbe40a16319c19a8a4d8cf9d9bb67aa0512f522e99289fd3495")

pro = ts.pro_api("d0099dbe40a16319c19a8a4d8cf9d9bb67aa0512f522e99289fd3495")

df = pro.query('index_daily',ts_code='000852.CSI', start_date='20100101')
df.to_csv('000852.csv')
# df = pd.read_csv()

# df = pro.trade_cal(exchange='', start_date='20180901', end_date='20181001', fields='exchange,cal_date,is_open,pretrade_date', is_open='0')