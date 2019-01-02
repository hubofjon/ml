# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 16:07:44 2018

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
from P_commons import read_sql, to_sql_replace, to_sql_append
import matplotlib.pyplot as plt

def plot_base(q_date, ticks='', do=''): #ticks =['x','y'], do(unop)
    import plotly.plotly as plty
    import plotly.graph_objs as plgo
    import matplotlib.pyplot as plt
    from matplotlib.finance import candlestick2_ohlc as cdl
    from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY
    import matplotlib.ticker as mticker
    import matplotlib.dates as mdates
    mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
    alldays = DayLocator()              # minor ticks on the days
    weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
    dayFormatter = DateFormatter('%d')      # e.g., 12
    import pylab
    import matplotlib
    matplotlib.use('TkAgg')
#    %matplotlib inline
    ticks = [x.upper() for x in ticks]
#    tickers=df['ticker'].values
#    tickers=list(set(tickers))  #get only unique list value
    p_date=q_date-datetime.timedelta(100)
#    qry_sp="SELECT * FROM tbl_pvt_sp500  wHERE date BETWEEN '%s' AND '%s'" %(p_date, q_date)
#    qry_etf="SELECT * FROM tbl_pvt_etf  wHERE date BETWEEN '%s' AND '%s'" %(p_date, q_date)
#    dp_etf=read_sql(qry_etf, q_date)
#    dp_sp=read_sql(qry_sp,q_date)
#    dp=pd.concat([dp_etf, dp_sp], axis=0)
    qry="SELECT * FROM tbl_pv_all  wHERE date BETWEEN '%s' AND '%s'" %(p_date, q_date)
    dp=read_sql(qry,q_date)
    pd.set_option('display.expand_frame_repr', False)
    for t in ticks:
        if t in dp.ticker.unique():
            dt=dp[dp.ticker == t]
            dt=dt.sort_values('date', ascending=True)
            dt=dt.tail(80)
            dt['date']=dt['date'].astype(str).apply(lambda x: x[:10])
            dt['date']=pd.to_datetime(dt['date'],format='%Y-%m-%d')# %H:%M:%S.%f')
    #        dt=dt[['date', 'close','volume']]
            dt['date']=dt['date'].map(mdates.date2num)
#            dt=dt.set_index('date')
    #        dt.plot(subplots=True, figsize=(10,4), rot=45)
    #        plt.legend(loc='best')
    #        plt.show()   
#            dt=dt.set_index('date')
#            fig, ax=plt.subplots(figsize=(16,3))
##            if (do.empty | len(do)==0):
#            if len(do)==0:
#                plt.title("%s:  "%(t))
#            else:
#                plt.title("%s:  "%(do[do.ticker==t]))
#
#            cdl(ax,dt['open'], dt['high'], dt['low'], dt['close'], width=0.6)
#            plt.show()
#            dt['volume']=dt['volume']/100000
#            dt['volume'].plot(rot=90, figsize=(15,2), kind='bar')
#            plt.show()
            dt=dt[['date','open','close','high','low','volume']]
            fig=plt.figure()
            ax1=plt.subplot2grid((5,4),(0,0),rowspan=4, colspan=4)
#            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#            ax1.xaxis.set_major_locator(mticker.MaxNLocator(60))
#            ax1.xaxis.set_major_locator(mondays)
#            ax1.xaxis.set_minor_locator(alldays)
#convert raw mdate number to dates
#            ax1.xaxis_date()
            plt.xlabel("date")
            cdl(ax1,dt['open'],dt['high'], dt['low'], dt['close'], width=0.6)
            plt.ylabel('stk price')
            plt.legend()

            plt.show()
#            ax1.grid(True)

#            
#            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
#            plt.setp(ax1.get_xticklabels(),visible=False)
#            
            ax2=plt.subplot2grid((5,4),(4,0), sharex=ax1, rowspan=1, colspan=4)
            ax2.bar(dt.index,dt['volume'])
            ax2.axes.yaxis.set_ticklabels([])
            plt.subplots_adjust(hspace=0)
#            ax2.grid(True)
#            plt.ylabel('Volume')
#            
#            for label in ax.xaxis.get_ticklablels():
#                label.set_rotatiion(45)
            plt.show()
            
#            fig.autofmt_xdate()
    pd.set_option('display.expand_frame_repr', True)      

#Spike profile, n=hv windows, s=sample days
def plot_HV(df='', ticks=''):
#sort or drop will return a new object to avoid mutation to original df as paramters    
    df=df.sort_values('date',ascending=True)
    buckets=[-4,-3,-2,-1,0,1,2,3,4]
    win_1=20
    win_2=60
    sap_1=1
    sap_2=7
#    df['date']=pd.to_datetime(df['date'])

#hv_profile by window    

    for w in [win_1, win_2]:
#        fig, ax=plt.subplots(figsize=(16,3))
        print("hv_profile_%s'"%w)
        df['log_rtn']=np.log(1+df['close'].pct_change())
        df['std']=df['log_rtn'].rolling(w).std()
        df['hv']=df['std']*(252**0.5)
        df['p_std']= df['close']*df['std']
#        df['spike']=(df['close']-df['close'].shift(1))/df['p_std'].shift(1)
        df.dropna(inplace=True)
        df.plot(x='date',y='std', kind='line', figsize=(15,4), rot=45)     
#spike_profile by sample days
    for s in [sap_1, sap_2]:
        print("spike_profile_%s'"%s)
        if s>1:   #resample
            df['date']=pd.to_datetime(df['date'])
            df.set_index('date',inplace=True)
            df=df.resample('W')
            df=df.reset_index(level='date')

#        fig, ax=plt.subplots(figsize=(16,3))
        df['log_rtn']=np.log(1+df['close'].pct_change())
        df['std']=df['log_rtn'].rolling(20).std()
        df['hv']=df['std']*(252**0.5)
        df['p_std']= df['close']*df['std']
        df['spike']=(df['close']-df['close'].shift(1))/df['p_std'].shift(1)
        df.dropna(inplace=True)

        df.plot(x='date',y='spike', kind='bar', figsize=(15,4), rot=45)
        plt.show()
        df['bucket'] = pd.cut(df['spike'], bins=buckets)
        df['spike'].plot(kind='hist',xticks=buckets)
        print(df.bucket.value_counts())
#        fig.autofmt_xdate()
#        df.plot(x='date',y='std', kind='line', figsize=(15,4), rot=45)     
#        fig.autofmt_xdate()  
        
#        
#        fig.autofmt_xdate()
#    
    
#one month spike profile
       