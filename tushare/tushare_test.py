import tushare as ts

print(ts.__version__)

if __name__ == '__main__':
    print(ts.__version__)
    ts.set_token("857fdbd057267f71f3d6a80017f0c4b1d2e51eb06f70c5fe50ec7a48")
    pro = ts.pro_api()
    # 获取上证指数的历史数据
    # df = pro.index_daily(ts_code='000001.SH', start_date='20230101', end_date='20231001')
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    print(df)



