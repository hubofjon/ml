"""
Created on Tue Aug 15 11:13:46 2017
purpose: update price from eoddata download or yahoo/google same day
Function: 
    1. update_price_eod  (download eoddata)
    2. eod_to_sql_append
    3. update_price_google (same day or hisotrical)
    4. update_price_today_yahoo
 df_bc.head(1)
Out[334]: 
   index ticker  price type  strike   oexp_dt    dte  bid  midpoint  ask  \
0      0    HTZ  16.24  Put    3.00  01/17/20 403.00 0.10      0.15 0.20   

   last      vol     oi  v_oi       iv      date  ba_pct bs sweep  
0  0.20 14500.00 214.00 67.76  108.70%  12/10/18    1.00  b     b  


bc. DTE (days to expire), midpoint, vol, bs, type

dpv.columns
Out[337]: Index(['date', 'ticker', 'close', 'high', 'low', 'open', 'volume'],

R_update_price (trig_run)
       
@author: jon
"""

import pandas as pd
import numpy as np
import datetime as datetime
import pandas_datareader.data as web
from pandas_datareader import data, wb
import sys, os
import sqlite3 as db
import pandas.io.sql as pd_sql

import P_commons
from P_commons import read_sql, to_sql_replace, to_sql_append
from P_options import get_option_simple
from P_intel import get_sis_sqz, get_earnings_ec
# parameters
capital=20000
DF_sp500=pd.read_csv('c:\\pycode\pyprod\constituents.csv')
df_etf=pd.read_csv('c:\\pycode\pyprod\etf.csv')
df_sp500=DF_sp500.iloc[:,0] #serie
df_etf=df_etf.iloc[:,0]

todate=datetime.date.today()

def update_price_eod(underlying):#
    if underlying =='sp500':
        df_symbol=df_sp500
    elif underlying =='etf':
        df_symbol=df_etf
    
    path='c:\\pycode\\eod'
    path=r"%s"%path
    files=os.listdir(path)
    #list csv files
    files_csv=[f for f in files if f[-3:]=='csv']
    if len(files_csv)==0:
        print("eod file not available")
        exit
    date_max=read_sql("SELECT max(date) FROM tbl_price", todate).iloc[0,0]

    df_price=pd.DataFrame()
    for f in files_csv:
        date_csv=f[4:12]
        if date_csv <=date_max:
            print("price of '%s' already exit"%date_csv)
            sys.exit()
        df=pd.read_csv(r'c:\pycode\eod\%s'%f)#, index_col=0)
# filter df with series df_sp500
        df=df[df['A'].isin(df_symbol)]
        df=df.iloc[:,[0,1,5]] #ticker, date, close
        df=df.T
        df=df.rename(columns=df.iloc[0]) #convert row to columns name
        df['date']=df.iloc[1,1]
        df['date']=pd.to_datetime(df['date'])  #convert string to datetime
#        df.set_index('date', inplace=True)
#        df.index.name='date'
        df=df.iloc[2,:] #keep close only
        df_price=df_price.append(df)
#        os.remove(r'c:\pycode\eod\%s'%f)
    df_price.reset_index(drop=True, inplace=True)
    df_price.sort_values('date', ascending=True, inplace=True) 
    eod_to_sql_append(df_price, underlying)
    return df_price

def eod_to_sql_append(df, underlying):
    conn=db.connect('c:\\pycode\db_op.db')
    
    if underlying =='sp500':
        pd_sql.to_sql(df, "tbl_price", conn, if_exists='append', index=False)
    elif underlying =='etf':
        pd_sql.to_sql(df, "tbl_price_etf", conn, if_exists='append', index=False)
    print("%s eod: %s appended to tbl_price"%(underlying, todate))


def update_pv_eod(underlying):  #can update multiple datefiles
    if underlying =='sp500':
        df_symbol=df_sp500
    elif underlying =='etf':
        df_symbol=df_etf
    date_max=read_sql("SELECT max(date) FROM tbl_pv_etf", todate).iloc[0,0]     
    path='c:\\pycode\\eod'
    path=r"%s"%path
    files=os.listdir(path)
    #list csv files
    files_csv=[f for f in files if f[-3:]=='csv']
    if len(files_csv)==0:
        print("eod file not available")
        exit
    df=pd.DataFrame()    
    for f in files_csv:
        date_csv=f[4:12]
        if date_csv <=date_max:
            print("price of '%s' already exit"%date_csv)
            sys.exit()
        df_tmp=pd.read_csv(r'c:\pycode\eod\%s'%f)#, index_col=0)
# filter df with series df_sp500
        if underlying !='all':  #if sp500 or etf
            df_tmp=df_tmp[df_tmp['A'].isin(df_symbol)]
#        df=df.append(df_tmp)
        df_tmp.columns=[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]    
        df=pd.concat([df,df_tmp], axis=0)
#    df['date']=pd.to_datetime(df['date'])
#    df.set_index('date', inplace=True)
    df['date']=pd.to_datetime(df['date']).dt.date # convert str to datetime
    df.sort_values(['date','ticker'], ascending=['False', 'True'], inplace=True)
    
    conn=db.connect('c:\\pycode\db_op.db')    
    pd_sql.to_sql(df, "tbl_pv_%s"%underlying, conn, if_exists='append', index=False)
    print("EOD update to tbl_pv_%s is done "%underlying)
    
#    try:     
#        data=WEB.get_quote_yahoo(df_symbol)
#        dt=data.T
#        today=datetime.date.today()
#        dt['date']=today
#        dt['date']=pd.to_datetime(dt['date'])
##        dt.set_index('date',inplace=True)
#        df_close=dt.iloc[2].to_frame().transpose()
#    except:
#        print("cannot retrieve data_yahoo")
#    conn=db.connect('c:\\pycode\db_op.db')
#    pd_sql.to_sql(df_close, "tbl_price", conn, if_exists='append', index=False)
#    print("toda: %s yahoo data appended"%today.date())
# Panel 
def trig_run(underlying):
    vj_thread=2.5  # volume change %
    pj_thread=0.05  #price change %
    query="SELECT * FROM tbl_pv_%s"%underlying
    df=read_sql(query, todate)
    df['v_mean_66']=df.groupby('ticker')['volume'].apply(lambda x: x.rolling(66).mean().shift(1))
    con_vj=df['volume']/ df['v_mean_66']>=vj_thread
    #label vj daily
    df.loc[con_vj, 'vj']=df['volume']/ df['v_mean_66']
    df['pj']=df.groupby('ticker')['close'].apply(lambda x: x.pct_change())
    con_pj=np.abs(df['pj'])<=pj_thread
    df.loc[con_pj,'pj']=np.nan
    df.sort_values(['date','ticker'], ascending=[False, True], inplace=True)
    df= trig_mark_new(df)
    to_sql_replace(df, "tbl_pvt_%s"%underlying)
    
    trigger=df[pd.notnull(df.trig)]
    if trigger.shape[0] <12:
        print(trigger[['ticker','date','trig', 'vj','pj']])
        trig_plot(df, underlying)
        trigger['date_vj']=trigger['date']
        trigger['date']=max(df.date)  #latest datetime from tbl_pv
        do=pd.DataFrame()
        if underlying=='sp500':
            for index, row in trigger.iterrows():
                ticker=row['ticker']
                try:
                    do_tmp=get_option_simple(ticker)
                    do=do.append(do_tmp)
                except:
                    pass
            trigger=trigger.merge(do, on='ticker', how='left')
        to_sql_append(trigger, "tbl_pvth_%s"%underlying)
                
    else:
        print ("check")
    
def trig_mark(df):
    np.seterr(invalid='ignore')
    num_etf=52  #total number of etf
    min_interval=5
    fake_thread=0.01 # within 1% of vj high or low
    df_vj=df[pd.notnull(df.vj)]  #only when vj
    loc_vj=df.groupby('ticker')['vj'].apply(lambda x: x.argmax())  #find index of last valid vj
#    loc_vj=df_vj.groupby('ticker')['vj'].apply(lambda x: x.argmax())  #find index of last valid vj
    loc_close_1=df.groupby('ticker')['close'].tail(1).index
#    df.loc[loc_close_1, 'vj_date']=df.loc[loc_vj,'vj'].values
    close_1=df.groupby('ticker')['close'].tail(1)
    close_2=df.groupby('ticker')['close'].tail(2).head(1)
    close_3=df.groupby('ticker')['close'].tail(3).head(1)
    close_n=df.groupby('ticker')['close'].tail(min_interval).head(1)
#first time two daysin a row breakup
    con_vj=pd.notnull(df.loc[loc_vj]['vj'])
    con_breakup=(close_1.values>=df.loc[loc_vj,'high'].values) &\
            (close_2.values>=df.loc[loc_vj,'high'].values) &\
            (close_3.values<df.loc[loc_vj,'high'].values)  & \
            con_vj.values
    con_breakdn=(close_1.values<=df.loc[loc_vj,'low'].values) & \
            (close_2.values<=df.loc[loc_vj,'low'].values) &\
            (close_3.values>df.loc[loc_vj,'high'].values) &\
            con_vj.values
            
    con_duration=np.abs(loc_vj.values - loc_close_1.values)>=num_etf * min_interval

    con_fakeup=(close_1.values<=df.loc[loc_vj,'high'].values) & \
            (close_1.values/df.loc[loc_vj,'high'].values >=(1-fake_thread)) &\
            (close_n.values<=df.loc[loc_vj,'close'].values) &\
            con_duration & \
            con_vj.values
    con_fakedn=(close_1.values>=df.loc[loc_vj,'low'].values) & \
            (close_1.values/df.loc[loc_vj,'low'].values <=(1+ fake_thread)) &\
            (close_n.values>=df.loc[loc_vj,'close'].values) & \
            con_duration  &\
            con_vj.values
        
    df.loc[con_breakup, 'trig']='bu' #breakup
    df.loc[con_breakdn, 'trig']='bd' #breakdown
    df.loc[con_fakeup, 'trig']='fu' #breakup
    df.loc[con_fakedn, 'trig']='fd' #breakdown   
   
    return df

def trig_plot(df, underlying):
#    import plotly.plotly as plty
#    import plotly.graph_objs as plgo
    df=df[pd.notnull(df.trig)]
    import matplotlib.pyplot as plt
    from matplotlib.finance import candlestick2_ohlc as cdl
    
    tickers=df['ticker'].values
    tickers=list(set(tickers))  #get only unique list value
    for t in tickers:
        try:
            trig=df[df.ticker==t]['trig']
            if underlying =='sp500':
                query="SELECT * FROM tbl_pvt_sp500 WHERE ticker='%s'"%t
            elif underlying =='etf':
                query="SELECT * FROM tbl_pvt_etf WHERE ticker='%s'"%t
            dt=read_sql(query, todate)
            dt=dt.sort_values('date', ascending=True)
            dt=dt.tail(80)
            dt['date']=dt['date'].astype(str).apply(lambda x: x[:10])
    #        dt=dt[['date', 'close','volume']]
    #        dt=dt.set_index('date')
    #        dt.plot(subplots=True, figsize=(10,4), rot=45)
    #        plt.legend(loc='best')
    #        plt.show()   
            dt=dt.set_index('date')
            fig, ax=plt.subplots(figsize=(16,3))
            plt.title("%s:  %s"%(t, trig))
            cdl(ax,dt['open'], dt['high'], dt['low'], dt['close'], width=0.6)
            plt.show()
            dt['volume']=dt['volume']/100000
            dt['volume'].plot(rot=90, figsize=(15,2), kind='bar')
            plt.show()
        except:
            pass

def trig_mark_new(df):
    np.seterr(invalid='ignore')
    num_etf=52  #total number of etf
    min_interval=10
    fake_thread=0.01 # within 1% of vj high or low
    df_vj=df[pd.notnull(df.vj)]  #only when vj
    loc_vj=df_vj.groupby('ticker')['vj'].apply(lambda x: x.argmax())  #find index of last valid vj
#    loc_vj=df_vj.groupby('ticker')['vj'].apply(lambda x: x.argmax())  #find index of last valid vj
# alternative: x2=x1.drop_duplicates(['ticker'], keep='first')

    
    mask=df.loc[loc_vj, 'ticker']
    dmask=mask.to_frame(name='ticker')
    df=df.merge(dmask, on='ticker', how='right')
#    df.loc[loc_close_1, 'vj_date']=df.loc[loc_vj,'vj'].values
    df=df.sort_values(['ticker','date'], ascending=[True, False])
    close_1=df.groupby('ticker')['close'].nth(1)
    close_2=df.groupby('ticker')['close'].nth(2)
    close_3=df.groupby('ticker')['close'].nth(3)
    close_n=df.groupby('ticker')['close'].nth(min_interval)
    loc_close=df.groupby('ticker')['close'].nth(0).index
#first time two daysin a row breakup
    con_vj=pd.notnull(df.loc[loc_vj]['vj'])
    
    con_breakup=(close_1.values>=df.loc[loc_vj,'high'].values) &\
            (close_2.values>=df.loc[loc_vj,'high'].values) &\
            (close_3.values<df.loc[loc_vj,'high'].values)  & \
            con_vj.values
    con_breakdn=(close_1.values<=df.loc[loc_vj,'low'].values) & \
            (close_2.values<=df.loc[loc_vj,'low'].values) &\
            (close_3.values>df.loc[loc_vj,'high'].values) &\
            con_vj.values
            
#    con_duration=np.abs(loc_vj.values - loc_close.values)>=num_etf * min_interval

    con_fakeup=(close_1.values<=df.loc[loc_vj,'high'].values) & \
            (close_1.values/df.loc[loc_vj,'high'].values >=(1-fake_thread)) &\
            (close_n.values<=df.loc[loc_vj,'close'].values)
#            con_duration & \
#            con_vj.values
    con_fakedn=(close_1.values>=df.loc[loc_vj,'low'].values) & \
            (close_1.values/df.loc[loc_vj,'low'].values <=(1+ fake_thread)) &\
            (close_n.values>=df.loc[loc_vj,'close'].values) 
#            con_duration  &\
#            con_vj.values
        
    df.loc[con_breakup, 'trig']='bu' #breakup
    df.loc[con_breakdn, 'trig']='bd' #breakdown
    df.loc[con_fakeup, 'trig']='fu' #breakup
    df.loc[con_fakedn, 'trig']='fd' #breakdown   
   
    return df






def vpj_plot(underlying):
    q1="SELECT max(date) FROM tbl_pvt_%s"%underlying
    dq1=read_sql(q1,todate)
    max_date=dq1.iloc[0,0]
    qry="SELECT * from tbl_pvt_%s where date='%s'"%(underlying, max_date)
    df=read_sql(qry,todate)
    vj=df[pd.notnull(df.vj)]
    if vj.shape[0]<10:
        print (vj[['ticker','vj','pj','trig', 'date']])
        vj.trig='vj'
        trig_plot(vj, underlying)
    pj=df[pd.notnull(df.pj)]
    if vj.shape[0]<10:
        print (pj[['ticker','vj','pj','trig', 'date']])  
        pj.trig='pj'
        trig_plot(pj, underlying)        

def trig_run_1(underlying):
    vj_thread=2.5  # volume change %
    pj_thread=0.05  #price change %
    query="SELECT * FROM tbl_pv_%s"%underlying
    df=read_sql(query, todate)
    df['v_mean_66']=df.groupby('ticker')['volume'].apply(lambda x: x.rolling(66).mean().shift(1))
    df['close_prev']=df.groupby('ticker')['close'].shift(1)

#    con_pchg=df['close']>df['close_prev']
    df['vj']=df['volume']/ df['v_mean_66']
    df['pj']=df.groupby('ticker')['close'].apply(lambda x: x.pct_change())
    df=df.dropna() 
    
    con_vj=df['volume']/ df['v_mean_66']>=vj_thread
    #label vj daily
#    df.loc[(con_vj & con_pchg), 'vj']=(df['volume']/ df['v_mean_66']) 
#    df.loc[(con_vj & (~con_pchg)), 'vj']=-(df['volume']/ df['v_mean_66'])     
    con_pj=np.abs(df['pj'])>=pj_thread
#    df.loc[con_pj,'pj']=np.nan
#### tests here    
#    con_vjup=df['close']>df['close_prev']
#    df['max_1m']=df.groupby('ticker')['close'].tail(20).max()
#    df_1m=df.groupby('ticker').head(22)
#    max_1m=df.groupby('ticker')['close'].max()
#    min_1m=df.groupby('ticker')['close'].min()
#    loc_0=df.groupby('ticker')['close'].tail(1).index
#    con_pup=df.loc[loc_0,'close']>df.loc[loc_0,'close_prev']
#    
#    con_max=(df.loc[loc_0].sort_values('ticker',ascending=True)['close'].values==max_1m.values)
    
#####test end    
    df.sort_values(['date','ticker'], ascending=[False, True], inplace=True)
    
    
    
    df= trig_mark_new_1(df)
    to_sql_replace(df, "tbl_pvt_%s"%underlying)
#    
    trigger=df[pd.notnull(df.trig)]
    if underlying =='sp500':
        trigger=get_sis_sqz(trigger)
        trigger=get_earnings_ec(trigger)
#    if trigger.shape[0] <12:
    print(trigger.trig.value_counts())
    print(trigger[['ticker','date','trig', 'vj','pj','close','close_prev']])
    if underlying =='sp500':
        print(trigger[['ticker', 'si', 'si_chg','pe', 'hi_1y_fm', 'ma_200_fm', 'ma_50_fm']])
        print(trigger[['ticker', 'earn','earn_time','st','it','lt','i_sent','a_sent']])
        print(trigger[['ticker','sec','ind','com']])
    trig_plot(df, underlying)
##        trigger['date_vj']=trigger['date']
#        trigger['date']=max(df.date)  #latest datetime from tbl_pv
#        do=pd.DataFrame()
#        if underlying=='sp500':
#            for index, row in trigger.iterrows():
#                ticker=row['ticker']
#                try:
#                    do_tmp=get_option_simple(ticker)
#                    do=do.append(do_tmp)
#                except:
#                    pass
#            trigger=trigger.merge(do, on='ticker', how='left')
#        to_sql_append(trigger, "tbl_pvth_%s"%underlying)
#                
#    else:
#        print ("check")

def trig_mark_new_1(df):
    np.seterr(invalid='ignore')
    num_etf=52  #total number of etf
    min_interval=5
    fake_thread=0.01 # within 1% of vj high or low
    vj_thread=2.5
    pj_thread=0.05
#    df_vj=df[pd.notnull(df.vj)]  #only when vj
    loc_vj=df[df.vj>vj_thread].groupby('ticker')['vj'].apply(lambda x: x.argmax())  #find index of last valid vj
#    loc_vj=df_vj.groupby('ticker')['vj'].apply(lambda x: x.argmax())  #find index of last valid vj
# alternative: x2=x1.drop_duplicates(['ticker'], keep='first')

    
    mask=df.loc[loc_vj, 'ticker']
    dmask=mask.to_frame(name='ticker')
    dm=df.merge(dmask, on='ticker', how='right')  #untouched df for series price
#    df.loc[loc_close_1, 'vj_date']=df.loc[loc_vj,'vj'].values
    dm=dm.sort_values(['ticker','date'], ascending=[True, False])
    close_0=dm.groupby('ticker')['close'].nth(0)
    close_1=dm.groupby('ticker')['close'].nth(1)
    close_2=dm.groupby('ticker')['close'].nth(2)
    close_3=dm.groupby('ticker')['close'].nth(3)
    close_n=dm.groupby('ticker')['close'].nth(min_interval)
    loc_close=dm.groupby('ticker')['close'].nth(0).index
#first time two daysin a row breakup
    con_vjup=df.loc[loc_vj]['pj']>0
    
    con_breakup=(close_0.values>=df.loc[loc_vj,'high'].values) &\
            (close_1.values<=df.loc[loc_vj,'high'].values) &\
            (close_2.values<df.loc[loc_vj,'high'].values)  & \
            con_vjup.values
    con_breakdn=(close_0.values<=df.loc[loc_vj,'low'].values) & \
            (close_1.values>=df.loc[loc_vj,'low'].values) &\
            (close_2.values>df.loc[loc_vj,'low'].values) &\
            (~con_vjup.values)
            
#    con_duration=np.abs(loc_vj.values - loc_close.values)>=num_etf * min_interval

    con_edgeup=(close_0.values<=df.loc[loc_vj,'high'].values) & \
            (close_0.values/df.loc[loc_vj,'high'].values >=(1-fake_thread)) &\
            (close_n.values<=df.loc[loc_vj,'close'].values)
#            con_duration & \
#            con_vj.values
    con_edgedn=(close_0.values>=df.loc[loc_vj,'low'].values) & \
            (close_0.values/df.loc[loc_vj,'low'].values <=(1+ fake_thread)) &\
            (close_n.values>=df.loc[loc_vj,'close'].values) 
#            con_duration  &\
#            con_vj.values

    con_fakeup=(close_0.values<=df.loc[loc_vj,'close_prev'].values) &\
        (close_1.values>=df.loc[loc_vj,'close_prev'].values) &\
            con_vjup.values
            
    con_fakedn=(close_0.values>=df.loc[loc_vj,'close_prev'].values) &\
        (close_1.values<=df.loc[loc_vj,'close_prev'].values) &\
            (~con_vjup.values)       
        
    df.loc[con_breakup, 'trig']='bu' #breakup

    df.loc[con_breakdn, 'trig']='bd' #breakdown
    df.loc[con_fakeup, 'trig']='fu' #breakup
    df.loc[con_fakedn, 'trig']='fd' #breakdown   
#    df.loc[con_edgeup, 'trig']='eu' #breakup
#    df.loc[con_edgedn, 'trig']='ed' #breakdown   

    return df


