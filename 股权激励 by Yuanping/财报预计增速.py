import numpy as np
import pandas as pd
import numpy as np
import os

file = pd.read_excel('profit_19-22/业绩预警2022.xlsx')
file = file[file['预告净利润最大变动幅度(%)']>=50]

df = pd.DataFrame(columns=['证券代码', '名称', '预告日期', '预警类型', '预警摘要', '预告净利润最大变动幅度(%)', '预告净利润下限(万元)',
       '预告净利润上限(万元)', '预告净利润同比增长下限(%)', '预告净利润同比增长上限(%)', '定期报告预计披露日期', '预警内容',
       '是否最新', '首次预告日期', '是否变脸', '报告期', '证监会行业', 'Wind行业'])

for file in os.listdir('profit_19-22'):
    data = pd.read_excel('profit_19-22/' + file)
    data = data[data['预告净利润最大变动幅度(%)']>=50]
    data.drop('序号',axis=1)
    df = pd.concat([df,data])

df = df.sort_values(by=['证券代码','预告日期'], ascending=True)
df.index = np.arange(0,len(df))

price_data = pd.read_csv('stock_data/000006.SZ.csv',index_col=0)
price_data.sort_values('trade_date', inplace=True)
part_df = df[df['证券代码']=='000006.SZ']

# df.to_csv('预计增速.csv')

