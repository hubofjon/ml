import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime, time
def import_tick():
    df=pd.read_excel(r'C:\Users\qli1\BNS_wspace\records.xlsx', sheetname="tick")
    df['date']=pd.to_datetime(df['date'], format='%Y%m%d')
    dtd=pd.DataFrame()
    dtd['open']=df.groupby(df['date']).first()['open']
    dtd['close']=df.groupby(df['date']).last()['close']
    dtd['low']=df.groupby(df['date'])['high'].max()
    dtd['high']=df.groupby(df['date'])['high'].max()
    dtd['low']=df.groupby(df['date'])['low'].max()
    dtd['volume']=df.groupby(df['date'])['volume'].sum()
    dtd['amt']=df.groupby(df['date'])['amt'].sum()
    dtd['p_pct_chg']=dtd['close']/dtd['close'].shift(1)-1
    dtd['v_pct_chg']=dtd['volume']/dtd['volume'].shift(1)-1
    dtd['atr_pct']=(dtd['high']-dtd['low'])/dtd['close'].shift(1)
    #dtd.drop(dtd.index[[0,0]], inplace=True)
    dtd['date']=df.groupby(df['date'])


    return dtd,df

def import_data():
    dfs=pd.DataFrame()
    df=pd.read_excel(r'C:\Users\qli1\BNS_wspace\records.xlsx', sheetname="data")
    df['pl']=-df['amt_dealt']*df['bss']
    df['position']=df['qty_dealt']*df['bss']
    outlier_position=['2018-04-19','2018-04-20','2017-11-16','2017-11-17']
    con_pos=df['date'].isin(outlier_position)
    df=df[~con_pos]
    df['datetime']=df['date'].astype(str)+ " "+ df['timestamp'].astype(str)
    df['datetime']=pd.to_datetime(df['datetime'])
    df['dts']=df['datetime'].dt.round('min')
    dfs['pl']=df.groupby(df['date'])['pl'].sum()
#    position_by_date=df.groupby(df['date'])['position'].sum()
    return df, dfs

def anomaly_check(df):
        print("total trade count:", df.shape[0])
        df['price_dealt_pctl']=(df['price_dealt']-df['low'])/(df['high']-df['low'])
        df[np.isinf(df['price_dealt_pctl'])]=np.nan
        df[np.isneginf(df['price_dealt_pctl'])]=np.nan
        inf_count=df[np.isnan(df['price_dealt_pctl'])].shape[0]
        print("inf_count:", inf_count)
#        df.dropna(inplace=True)
        con_odd=(df['price_dealt_pctl']<0) | (df['price_dealt_pctl']>1)
        con_b_odd=(df['bss']==1) & con_odd
        con_s_odd=(df['bss']==(-1)) &  con_odd
        b_pctl_odd=df[con_b_odd][['date','timestamp','price_dealt', 'bs', 'high','low']]
        s_pctl_odd=df[con_s_odd][['date','timestamp','price_dealt','bs', 'high','low']]
        print("number of odd price:", df[con_odd].shape[0])
        print("number of normal price:", df[~con_odd].shape[0])
        print("number of buy price anamoly:  ", b_pctl_odd.shape[0])
        print("number of sell price anamoly:  ", s_pctl_odd.shape[0])
#        print(b_pctl_odd)
#        print(s_pctl_odd)
        con_b=(df['bss']==1) & (~con_odd)
        con_s=(df['bss']==(-1)) & (~con_odd)
        b_pctl=df[con_b]
        s_pctl=df[con_s]
        print("b_pctl stat: \n", b_pctl['price_dealt_pctl'].describe())

        b_pctl['price_dealt_pctl'].plot.hist(bins=10, alpha=0.2)


        print("s_pctl stat: \n ", s_pctl['price_dealt_pctl'].describe())
        s_pctl['price_dealt_pctl'].plot.hist(bins=10, alpha=0.2)
        plt.show()

        return dt
    
dtd, dt=import_tick()
df,dfs=import_data()    
anomaly_check(df)     

#pl_by_date.cumsum().plot()

