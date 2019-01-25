# -*- coding: utf-8 -*-
"""
KEY source:
https://realpython.com/python-matplotlib-guide/
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
import matplotlib.pyplot as plt

#Spike profile, n=hv windows, s=sample days

def read_sql(query):
    conn=db.connect('C:\\Users\qli1\BNS_git\git.db')
    df=pd_sql.read_sql(query, conn)
    conn.close()
    return df
df=read_sql("SELECT * FROM tbl_pv_etf" )
df=df[df.ticker=='SPY']
df.sort_values('date',ascending=True, inplace=True)
df.dropna(inplace=True)
df=df.tail(250)
def plot_t(df):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec   
    buckets=[-4,-3,-2,-1,0,1,2,3,4]
    
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(15, 12))
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = ax.flatten()
    
    df['log_rtn']=np.log(1+df['close'].pct_change())
    
    df['std_20']=df['log_rtn'].rolling(20).std()
    df['hv_20']=df['std_20']*(252**0.5)
    df['p_std_20']= df['close']*df['std_20']
    df['spike_1d']=(df['close']-df['close'].shift(1))/df['p_std_20'].shift(1)


#    b_count=df.bucket.value_counts()
#    b_pct=df.bucket.value_counts(normalize=True)
#    bucket_view=pd.concat([b_count,b_pct], axis=1, keys=['counts', '%'])
# #       df['bucket_pct']=df['bucket']/(df['bucket'].sum())
#    title_spike_dist="spike_distribution_%s"%sap
#    df['spike'].plot(kind='hist',xticks=buckets, title=title_spike_dist)

    df['std_60']=df['log_rtn'].rolling(60).std()
    df['hv_60']=df['std_60']*(252**0.5)
    df['p_std_60']= df['close']*df['std_60']
    
    df.dropna(inplace=True)
    
#ax1: hv_20_60
    df[['hv_20','hv_60']].plot(kind='line', title='hv', ax=ax1)
#    ax.set_xlim(xmin=df['date'][0], xmax=df['date'][-1])
    ax1.legend(loc='upper left')
#    fig.tight_layout()

#ax2: 
    
#ax3: spike_1d    
    df.plot(x='date',y='spike_1d', kind='bar', title="spike_1d", ax=ax7)# rot=45, ax=ax3)
    

#ax5: spike_1d_distribution
    df['spike_1d'].plot(kind='hist',xticks=buckets, title="spkike_1d_distribution", ax=ax5)

#ax7: spike_1d_dist_stat
    
    df['bucket_1d'] = pd.cut(df['spike_1d'], bins=buckets)
    b_count_1d=df.bucket_1d.value_counts()
    b_pct_1d=df.bucket_1d.value_counts(normalize=True)
    bucket_view_1d=pd.concat([b_count_1d,b_pct_1d], axis=1, keys=['count_1d', '%'])
    ax7.text(1,1,"bucket_view_1d")
    
#ax8:
    data = [[ 66386, 174296,  75131, 577908,  32015],\
        [ 58230, 381139,  78045,  99308, 160454],\
        [ 89135,  80552, 152558, 497981, 603535],\
        [ 78415,  81858, 150656, 193263,  69638],\
        [139361, 331509, 343164, 781380,  52269]]
#    plt.table(cellText=data, loc='bottom', ax=ax8)  
    ax8.table(cellText=data, loc='bottom')
    plt.show()
    return df
   
    
def plot_grid():
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec


    def make_ticklabels_invisible(fig):
        for i, ax in enumerate(fig.axes):
            ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
            ax.tick_params(labelbottom=False, labelleft=False)

    f = plt.figure(figsize=(20,24))
   
    ax_hv = plt.subplot2grid((4,4),(0,0),colspan=2)
    ax_hvc = plt.subplot2grid((4,4),(0,2),colspan=2)
    ax_k1 = plt.subplot2grid((4,4),(1,0),colspan=2)
    ax_k2 = plt.subplot2grid((4,4),(1,2),colspan=2)
    ax_d1 = plt.subplot2grid((4,4),(2,0),colspan=2)
    ax_d2 = plt.subplot2grid((4,4),(2,2),colspan=2)
    ax_s1 = plt.subplot2grid((4,4),(3,0),colspan=2)
    ax_s2 = plt.subplot2grid((4,4),(3,2),colspan=2)
   
    make_ticklabels_invisible(f)
#    plt.show()
    axes=[ax_hv, ax_hvc, ax_k1, ax_k2, ax_d1, ax_d2, ax_s1, ax_s2]
    return axes
    
def plot_HV(DF='', ticks=''):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    def make_ticklabels_invisible(fig):
        for i, ax in enumerate(fig.axes):
            ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
            ax.tick_params(labelbottom=False, labelleft=False)

    f = plt.figure(figsize=(20,24))
   
    ax_hv = plt.subplot2grid((4,4),(0,0),colspan=2)
    ax_hvc = plt.subplot2grid((4,4),(0,2),colspan=2)
    ax_k1 = plt.subplot2grid((4,4),(1,0),colspan=2)
    ax_k2 = plt.subplot2grid((4,4),(1,2),colspan=2)
    ax_d1 = plt.subplot2grid((4,4),(2,0),colspan=2)
    ax_d2 = plt.subplot2grid((4,4),(2,2),colspan=2)
    ax_s1 = plt.subplot2grid((4,4),(3,0),colspan=2)
    ax_s2 = plt.subplot2grid((4,4),(3,2),colspan=2)
   
    make_ticklabels_invisible(f)
#sort or drop will return a new object to avoid mutation to original df as paramters    
    DF=DF.sort_values('date',ascending=True)
    
    buckets=[-4,-3,-2,-1,0,1,2,3,4]

    sap_1=1    #resample_freq_1
    sap_2=7


#1. ax_hv (20)
    df=DF
#    df['log_rtn']=np.log(1+df['close'].pct_change())
    df['log_rtn']=np.log(df['close']/df['close'].shift(-1))
    
    df['std_20']=df['log_rtn'].rolling(20).apply(np.std())
    df['hv_20']=df['std']*(252**0.5)
    df['p_std_20']= df['close']*df['std']
    
    df['std_60']=df['log_rtn'].rolling(60).apply(np.std())
    df['hv_60']=df['std']*(252**0.5)
    df['p_std_60']= df['close']*df['std']
    
    df.dropna(inplace=True)
    title_hv="hv_20_60"
    df.plot(x='date',y=['std_20', 'std_60'], kind='line', title=title_hv, rot=45, ax=ax_hv, stacked=True)   
    plt.show()
    
#spike_profile by sample days
    for sap in [sap_1, sap_2]:
        if sap>1:   #resample
            df['date']=pd.to_datetime(df['date'])
            df.set_index('date',inplace=True)
            df=df.resample('W')
            df=df.reset_index(level='date')

        df['log_rtn']=np.log(1+df['close'].pct_change())
        df['std']=df['log_rtn'].rolling(20).std()
        df['hv']=df['std']*(252**0.5)
        df['p_std']= df['close']*df['std']
        df['spike']=(df['close']-df['close'].shift(1))/df['p_std'].shift(1)
        df.dropna(inplace=True)
        title_spike="spike_sample_%s"%sap
        df.plot(x='date',y='spike', kind='bar', title=title_spike, figsize=(15,4), rot=45)
        plt.show()
        df['bucket'] = pd.cut(df['spike'], bins=buckets)
        b_count=df.bucket.value_counts()
        b_pct=df.bucket.value_counts(normalize=True)
        bucket_view=pd.concat([b_count,b_pct], axis=1, keys=['counts', '%'])
 #       df['bucket_pct']=df['bucket']/(df['bucket'].sum())
        title_spike_dist="spike_distribution_%s"%sap
        df['spike'].plot(kind='hist',xticks=buckets, title=title_spike_dist)
#        print(bucket_view)
#        plt.show()
##        fig.autofmt_xdate()

def normal_test(df_test):
    #shapiro, sample size <5000
    import scipy.stats as stats
    statistic, pvalue = stats.shapiro(df_test)
    print("pvalue;", pvalue, "normal? :", pvalue>0.05 )
    #qq_plot
    import statsmodels.api as sm    
    sm.qqplot(df_test, line='s')
#https://medium.com/@rrfd/testing-for-normality-applications-with-python-6bf06ed646a9
# shapre understand: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3693611/   
    
#one month spike profile
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
