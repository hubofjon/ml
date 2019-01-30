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
import matplotlib.gridspec as gridspec   
import warnings
warnings.filterwarnings("ignore")

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
df=df.tail(310)
df=df.reset_index()

def plot_spikes(q_date, spec_list=['']):
    p_date=q_date - datetime.timedelta(350)
    spec_tuple=tuple(spec_list)
    if len(spec_list)>1:
        qry="SELECT * FROM tbl_pv_etf where ticker in %s"%(spec_tuple,)
    else:
        qry="SELECT * FROM tbl_pv_etf where ticker = '%s'"%(spec_tuple)
    dp=read_sql(qry)
    dp['date']=pd.to_datetime(dp['date'])
    dp=dp[dp.date>p_date]
    
    for t in spec_list:
        df=dp[dp.ticker==t]
        df.sort_values('date', ascending=True, inplace =True)
        print(" -- plot spike of %s --\n\n" %t)
        plot_spike(df)
    print(" plot_spikes completed" )
        
def plot_spike(df):
#https://realpython.com/python-matplotlib-guide/
#Data prepare
    s_interval=7 #spike_interval
    
    df['log_rtn_d']=np.log(1+df['close'].pct_change())
    df['std_20']=df['log_rtn_d'].rolling(20).std()
    df['hv_20']=df['std_20']*(252**0.5)
    df['p_std_20']= df['close']*df['std_20']
    df['spike_d']=(df['close']-df['close'].shift(1))/df['p_std_20'].shift(1)
    df['std_60']=df['log_rtn_d'].rolling(60).std()
    df['hv_60']=df['std_60']*(252**0.5)
    df['p_std_60']= df['close']*df['std_60']
#spike profile@s_interval days   
    df['log_rtn_s']=np.log(1+df['close'].pct_change(periods=s_interval))
    df['std_s']=df['log_rtn_s'].rolling(20).std()
    df['p_std_s']= df['close']*df['std_s']    
    df['spike_s']=(df['close']-df['close'].shift(s_interval))/df['p_std_s'].shift(1)
    df.dropna(inplace=True)
    
#PLOT 
    fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(15, 20))
    plt.subplots_adjust(hspace=0.5)
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12 = ax.flatten()
    fig.autofmt_xdate()
    fig.tight_layout()
    buckets=[-4,-3,-2,-1,0,1,2,3,4]    
    
#ax1: hv_20_60
    ax1=make_date(ax1)
    df[['hv_20','hv_60']].plot(kind='line', title='hv', ax=ax1)
#    ax.set_xlim(xmin=df['date'][0], xmax=df['date'][-1])
    ax1.legend(loc='upper left')

#ax2: 
    
#ax3: spike_d profile
    ax3=make_date(ax3)
    df.plot(x='date',y='spike_d', kind='bar', title="spike_d", ax=ax3)# rot=45, ax=ax3)

#ax4: spike_s profile
    ax4=make_date(ax4)
    df.plot(x='date',y='spike_s', kind='bar', title="spike_s", ax=ax4)# rot=45, ax=ax3)
    
#ax5: norm_d
    p_value_d=norm_test(df['log_rtn_d'],ax5)
    norm_d= p_value_d > 0.05

#ax6: stat_s & norm
    p_value_s=norm_test(df['log_rtn_s'],ax6)
    norm_s=p_value_s > 0.05
    
#ax7: dist_d & stat_d
    df['buck_d'] = pd.cut(df['spike_d'], bins=buckets)
    buck_cnt_d=df['buck_d'].value_counts()
    buck_pct_d=(df['buck_d'].value_counts(normalize=True)).map(lambda x: '{:,.0%}'.format(x))
    df_stat_d=pd.concat([buck_cnt_d,buck_pct_d], axis=1, keys=['count_d', '%'])
#    df['spike_d'].plot(kind='hist',xticks=buckets, ax=ax7)
    ax7=plot_blend(df['spike_d'], buckets, ax7, 'yellow')

    
#ax8: dist_s & stat_s
    df['buck_s'] = pd.cut(df['spike_s'], bins=buckets)
    buck_cnt_s=df['buck_s'].value_counts()
    buck_pct_s=(df['buck_s'].value_counts(normalize=True)).map(lambda x: '{:,.0%}'.format(x))
    df_stat_s=pd.concat([buck_cnt_s,buck_pct_s], axis=1, keys=['count_s', '%'])
    ax8=plot_blend(df['spike_s'], buckets, ax8, 'orange')

#ax9: stat_d_table
    ax9.axis('off')
    ax9.table(cellText=df_stat_d.values, \
              rowLabels=df_stat_d.index,\
              colLabels= df_stat_d.columns,\
              loc='center')
    
#ax10: stat_s_table
    ax10.axis('off')
    ax10.table(cellText=df_stat_s.values, \
              rowLabels=df_stat_s.index,\
              colLabels= df_stat_s.columns,\
              loc='center')
    
#ax11: next spike_d date>sig_std   
    sig=2
    con_sig_d=np.abs(df['spike_d'])>=sig
    dd=df[con_sig_d]
    dd['idx']=dd.index
    dd['idx_diff']=dd['idx'].diff()
    d=dd['idx_diff'].describe()
    dd['date']=pd.to_datetime(dd['date']).dt.date
    last_spike_d=dd.tail(1).date.values[0]
    d_25=last_spike_d + datetime.timedelta(int(d[4]))
    d_50=last_spike_d + datetime.timedelta(int(d[5]))
    d_75=last_spike_d + datetime.timedelta(int(d[6]))
    table_sig_d=pd.DataFrame([d_25, d_50, d_75], \
            columns=[' '],\
            index=['d_25','d_50','d_75'])
    title_sig_d="next_spike_d date > %s_std"%sig
#    dd.plot('idx_diff', kind='hist', ax=ax11, title=title_sig_d, legend=False)
    dd['idx_diff'].hist(ax=ax11, grid=False)
    ax11.set_title(title_sig_d)
    ax11.table(cellText=table_sig_d.values, \
              rowLabels=table_sig_d.index,\
              colLabels= table_sig_d.columns,\
              loc='bottom')
    
#ax12: next spike_s date>sig_std   
    con_sig_s=np.abs(df['spike_s'])>=sig
    ds=df[con_sig_s]
    ds['idx']=ds.index
    ds['idx_diff']=ds['idx'].diff()
    s=ds['idx_diff'].describe()
    ds['date']=pd.to_datetime(ds['date']).dt.date
    last_spike_s=ds.tail(1).date.values[0]
    s_25=last_spike_s + datetime.timedelta(int(s[4]))
    s_50=last_spike_s + datetime.timedelta(int(s[5]))
    s_75=last_spike_s + datetime.timedelta(int(s[6]))
    table_sig_s=pd.DataFrame([s_25, s_50, s_75], \
            columns=[' '],\
            index=['s_25','s_50','s_75'])
    title_sig_s="next spike_s date > %s_std"%sig
#    ds.plot('idx_diff', kind='hist', ax=ax12, title=title_sig_s, legend=False)
    ds['idx_diff'].hist(ax=ax12, grid=False)
    ax12.set_title(title_sig_s)
    ax12.table(cellText=table_sig_s.values, \
              rowLabels=table_sig_s.index,\
              colLabels= table_sig_s.columns,\
              loc='bottom')
 
    plt.show()
    plt.close('all')
#    return df, dd, ds
#    plt.subplots_adjust(left=0.5)

def make_date(ax):
    from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MonthLocator, MONDAY, FRIDAY
    import matplotlib.ticker as mticker
    import matplotlib.dates as mdates
#    mondays = WeekdayLocator(FRIDAY)        # major ticks on the mondays
#    alldays = DayLocator()              # minor ticks on the days
#    fig, ax = plt.subplots(figsize=(12, 6))
#    fig.autofmt_xdate()
 #   ax.plot_date(df['date'], df['volume'], ls='', marker='x')
#    ax.xaxis.set_major_locator(mticker.MaxNLocator(60))
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_minor_locator(WeekdayLocator(MONDAY))
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax.xaxis.set_minor_formatter(DateFormatter('%d'))
    ax.fmt_xdata = DateFormatter('%d')
    return ax

def norm_test(serie, ax):
    #shapiro, sample size <5000
    import scipy.stats as stats
    import statsmodels.api as sm
    statistic, p_value = stats.shapiro(serie)
    result=p_value>0.05
    title_norm="log_rtn norm?: %s"%(result)
    sm.qqplot(serie, line='s', ax=ax)
    ax.set_title(title_norm)
    return p_value
#https://medium.com/@rrfd/testing-for-normality-applications-with-python-6bf06ed646a9
# shapre understand: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3693611/   
def plot_blend(serie, buckets,ax, color):
    import matplotlib.transforms as transforms
    import matplotlib.patches as mpatches
 #   fig, ax = plt.subplots()
#    x = np.random.randn(1000)
#    ax.hist(x, 30)
#    ax.set_title(r'$\sigma=1 \/ \dots \/ \sigma=2$', fontsize=16)
    # the x coords of this transformation are data, and the
    # y coord are axes
    serie.plot(kind='hist',xticks=buckets, ax=ax)
    trans = transforms.blended_transform_factory(
        ax.transData, ax.transAxes)
    
    # highlight the 1..2 stddev region with a span.
    # We want x to be in data coordinates and y to
    # span from 0..1 in axes coords
    rect1 = mpatches.Rectangle((-2, 0), width=1, height=1,
                             transform=trans, color=color,
                             alpha=0.5)
    rect2 = mpatches.Rectangle((1, 0), width=1, height=1,
                             transform=trans, color=color,
                             alpha=0.5)    
    rect3 = mpatches.Rectangle((-4, 0), width=1, height=1,
                             transform=trans, color='red',
                             alpha=0.5)       
    ax.add_patch(rect1)
    ax.add_patch(rect2)    
#    plt.show()
    return ax

def plot_grid(DF='', ticks=''):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    def make_ticklabels_invisible(fig):
        for i, ax in enumerate(fig.axes):
            ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
            ax.tick_params(labelbottom=False, labelleft=False)
    f = plt.figure(figsize=(20,24))
    ax_hv = plt.subplot2grid((4,4),(0,0),colspan=2)
    make_ticklabels_invisible(f)

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
    import matplotlib
    ticks = [x.upper() for x in ticks]
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
            dt['date']=dt['date'].map(mdates.date2num)
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
#            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#            plt.setp(ax1.get_xticklabels(),visible=False)
            ax2=plt.subplot2grid((5,4),(4,0), sharex=ax1, rowspan=1, colspan=4)
            ax2.bar(dt.index,dt['volume'])
            ax2.axes.yaxis.set_ticklabels([])
            plt.subplots_adjust(hspace=0)
#            ax2.grid(True)
#            plt.ylabel('Volume')
#            for label in ax.xaxis.get_ticklablels():
#                label.set_rotatiion(45)
            plt.show()
    pd.set_option('display.expand_frame_repr', True)      
