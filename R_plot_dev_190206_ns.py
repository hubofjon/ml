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
#from P_commons import read_sql
warnings.filterwarnings("ignore")

#Spike profile, n=hv windows, s=sample days
#plot_fulls/ full, plot_pv (basic plot)
def read_sql(query):
    conn=db.connect('C:\\Users\qli1\BNS_git\git.db')
    df=pd_sql.read_sql(query, conn)
    conn.close()
    return df
df=read_sql("SELECT * FROM tbl_pv_all" )
df=df[df.ticker=='GS']
df.sort_values('date',ascending=True, inplace=True)
df.dropna(inplace=True)
df=df.tail(310)
df=df.reset_index()

def plot_alls(q_date, spec_list=[''], mode='semi'):
    p_date=q_date - datetime.timedelta(350)
    spec_tuple=tuple(spec_list)
    if len(spec_list)>1:
        qry="SELECT * FROM tbl_pv_all where ticker in %s"%(spec_tuple,)
    else:
        qry="SELECT * FROM tbl_pv_all where ticker = '%s'"%(spec_tuple)
    dp=read_sql(qry, q_date)
    dp['date']=pd.to_datetime(dp['date'])
    dp=dp[dp.date>p_date]
    for t in spec_list:
        df=dp[dp.ticker==t]
        df.sort_values('date', ascending=True, inplace =True)
        print(" -- plot spike of %s --\n\n" %t)
        plot_all(q_date, df, mode )
    print(" plot_spikes completed" )
        
def plot_all(q_date, df, mode='semi'):
#plot_blend, plot_pv, norm_test, p_mdate
#https://realpython.com/python-matplotlib-guide/
#Data prepare
    s_interval=20 #spike_interval: default 1 month
    df=df.sort_values('date')
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
    df.reset_index(inplace=True)
#PLOT 
    if mode =='full':
        fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(15, 20))
        ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12 = ax.flatten()
    else:
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(14, 8))
        ax1, ax2, ax7, ax8, ax11, ax12 = ax.flatten()
    plt.subplots_adjust(hspace=0.5)
    fig.autofmt_xdate()
    fig.tight_layout()
    buckets=[-4,-3,-2,-1,0,1,2,3,4]   
#ax1: hv_20_60
    ax1=p_mdate(ax1)
    df[['hv_20','hv_60']].plot(kind='line', title='hv', ax=ax1)
#    ax.set_xlim(xmin=df['date'][0], xmax=df['date'][-1])
    ax1.legend(loc='upper left')
#ax2: pv_ma
    ax2=p_mdate(ax2)
    plt.sca(ax2)
    plot_pv(q_date, df, mode='semi')
#ax3: spike_d profile
    if mode =='full':
        ax3=p_mdate(ax3)
        df.plot(x='date',y='spike_d', kind='bar', title="spike_d", ax=ax3)# rot=45, ax=ax3)
#ax4: spike_s profile
    if mode =='full':
        ax4=p_mdate(ax4)
        df.plot(x='date',y='spike_s', kind='bar', title="spike_s", ax=ax4)# rot=45, ax=ax3)
#ax5: norm_d
    import scipy.stats as stats
    statistic_d, p_value_d = stats.shapiro(df['log_rtn_d'])
    norm_d=p_value_d>0.05
    if mode =='full':
        norm_test(df['log_rtn_d'],ax5, mode)
#ax6: stat_s & norm
#    p_value_s=norm_test(df['log_rtn_s'],ax6, mode)
    statistic_s, p_value_s = stats.shapiro(df['log_rtn_s'])
    norm_s=p_value_s > 0.05
    if mode =='full':
        norm_test(df['log_rtn_s'],ax6, mode)
#ax7: dist_d & stat_d
    df['buck_d'] = pd.cut(df['spike_d'], bins=buckets)
    buck_cnt_d=df['buck_d'].value_counts()
    bd= (df['buck_d'].value_counts(normalize=True))  
    
    bd.sort_index(inplace=True)
    s1_d=bd[0]+bd[1]
    s2_d=bd[2]+ bd[-1]
    s3_d=bd[3]+ bd[4]+ bd[-2]+bd[-3]
    tbl_stat_d=pd.DataFrame([s1_d, s2_d, s3_d], columns=['pct_dist'], index=['s1_d', 's2_d', 's3_d'])
    tbl_stat_d.reset_index(level=0, inplace=True)
    buck_pct_d=(df['buck_d'].value_counts(normalize=True)).map(lambda x: '{:,.0%}'.format(x))
    df_stat_d=pd.concat([buck_cnt_d,buck_pct_d], axis=1, keys=['count_d', 'pct'])
    title_d="norm? %s"%norm_d
    plot_blend(df['spike_d'], buckets, title_d, ax7, 'yellow')
#ax8: dist_s & stat_s
    df['buck_s'] = pd.cut(df['spike_s'], bins=buckets)
    buck_cnt_s=df['buck_s'].value_counts()
    bs= (df['buck_s'].value_counts(normalize=True))  
    bs.sort_index(inplace=True)
    s1_s=bs[0]+bs[1]
    s2_s=bs[2]+ bs[-1]
    s3_s=bs[3]+ bs[4]+ bs[-2]+bs[-3]
    tbl_stat_s=pd.DataFrame([s1_s, s2_s, s3_s], columns=['pct_dist'], index=['s1_s', 's2_s', 's3_s'])
    tbl_stat_s.reset_index(level=0, inplace=True)    
    buck_pct_s=(df['buck_s'].value_counts(normalize=True)).map(lambda x: '{:,.0%}'.format(x))
    df_stat_s=pd.concat([buck_cnt_s,buck_pct_s], axis=1, keys=['count_s', 'pct'])
    title_s="norm? %s"%norm_s
    plot_blend(df['spike_s'], buckets, title_s, ax8, 'orange')
#ax9: stat_d_table
    if mode =='full':
        ax9.axis('off')
        ax9.table(cellText=df_stat_d.values, \
              rowLabels=df_stat_d.index,\
              colLabels= df_stat_d.columns,\
              loc='center')
#ax10: stat_s_table
    if mode =='full':
        ax10.axis('off')
        ax10.table(cellText=df_stat_s.values, \
              rowLabels=df_stat_s.index,\
              colLabels= df_stat_s.columns,\
              loc='center')
#ax11: next spike_d date>sig_std   
    sig=3
    con_sig_d=np.abs(df['spike_d'])>=sig
    dd=df[con_sig_d]
    dd['idx']=dd.index
    dd['idx_diff']=dd['idx'].diff()
    d=dd['idx_diff'].describe()
    #no count for sig>3
    if d[0]==0:
      d[4],d[5],d[6]=0,0,0  
    dd['date']=pd.to_datetime(dd['date']).dt.date
    if not dd.empty:
        last_spike_d=dd.tail(1).date.values[0]  
    else:
        last_spike_d=datetime.datetime(2018,1,1).date()    
    d_25=last_spike_d + datetime.timedelta(int(d[4]))
    d_50=last_spike_d + datetime.timedelta(int(d[5]))
    d_75=last_spike_d + datetime.timedelta(int(d[6]))
    df_spk_d=pd.DataFrame([d_25, d_50, d_75], \
            columns=['pct_spike'],\
            index=['d_25','d_50','d_75'])
#merge with tbl_stat_d
    tbl_spk_d=df_spk_d.reset_index(level=0)
    tbl_spk_d=pd.concat([tbl_spk_d, tbl_stat_d], axis=1)
    tbl_spk_d['pct_dist']=tbl_spk_d['pct_dist'].map(lambda x: '{:,.0%}'.format(x))
    if not dd.empty:
        if dd.shape[0]==1: 
        #only one occurence
            title_spk_d="ONLY one occurence: spike_d date > %s_std"%sig
        else:
            title_spk_d="next_spike_d date > %s_std"%sig
        dd.plot('idx_diff', kind='hist', ax=ax11, title=title_spk_d, legend=False)
    else:
        title_spk_d="no previous spike > %s_std"%sig
#    dd['idx_diff'].hist(ax=ax11, grid=False)
    ax11.set_title(title_spk_d)
    ax11.table(cellText=tbl_spk_d.values, \
              rowLabels=tbl_spk_d.index,\
              colLabels= tbl_spk_d.columns,\
              loc='bottom')
#ax12: next spike_s date>sig_std   
    con_sig_s=np.abs(df['spike_s'])>=sig
    ds=df[con_sig_s]
    ds['idx']=ds.index
    ds['idx_diff']=ds['idx'].diff()
    s=ds['idx_diff'].describe()
    #no count for sig>3
    if s[0]==0:
      s[4],s[5],s[6]=0,0,0  
    ds['date']=pd.to_datetime(ds['date']).dt.date
    if not ds.empty:
        last_spike_s=ds.tail(1).date.values[0]
    else:
        last_spike_s=datetime.datetime(2018,1,1).date()
    s_25=last_spike_s + datetime.timedelta(int(s[4]))
    s_50=last_spike_s + datetime.timedelta(int(s[5]))
    s_75=last_spike_s + datetime.timedelta(int(s[6]))
    df_spk_s=pd.DataFrame([s_25, s_50, s_75], \
            columns=[' '],\
            index=['s_25','s_50','s_75'])
#merge with tbl_stat_s
    tbl_spk_s= df_spk_s.reset_index(level=0)
    tbl_spk_s=pd.concat([tbl_spk_s, tbl_stat_s], axis=1)
    tbl_spk_s['pct_dist']=tbl_spk_s['pct_dist'].map(lambda x: '{:,.0%}'.format(x))
    if not ds.empty:
        if ds.shape[0]==1: 
        #only one occurence
            title_spk_s="ONLY one occurence: spike_s date > %s_std"%sig
        else:
            title_spk_s="next spike_s date > %s_std"%sig
        ds.plot('idx_diff', kind='hist', ax=ax12, title=title_spk_s, legend=False)
    else:
        title_spk_s="no previous spike > %s_std"%sig
#    ds['idx_diff'].hist(ax=ax12, grid=False)
    ax12.set_title(title_spk_s)
    ax12.table(cellText=tbl_spk_s.values, \
              rowLabels=tbl_spk_s.index,\
              colLabels= tbl_spk_s.columns,\
              loc='bottom')
    plt.show()
    plt.close('all')

def norm_test(serie, mode, ax):
    #series = df['log_rtn'], shapiro, sample size <5000
    import scipy.stats as stats
    import statsmodels.api as sm
    statistic, p_value = stats.shapiro(serie)
    result=p_value>0.05
    title_norm="log_rtn norm?: %s"%(result)
    if mode =='full':
        sm.qqplot(serie, line='s', ax=ax)
        ax.set_title(title_norm)
#https://medium.com/@rrfd/testing-for-normality-applications-with-python-6bf06ed646a9
# shapre understand: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3693611/   
#https://matplotlib.org/tutorials/introductory/lifecycle.html#the-lifecycle-of-a-plot

def plot_blend(serie, buckets, title, ax, color):
    import matplotlib.transforms as transforms
    import matplotlib.patches as mpatches
 #   fig, ax = plt.subplots()
#    x = np.random.randn(1000)
#    ax.hist(x, 30)
    ax.set_title(title)
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
    rect4 = mpatches.Rectangle((3, 0), width=1, height=1,
                             transform=trans, color='red',
                             alpha=0.5) 
    ax.add_patch(rect1)
    ax.add_patch(rect2)  
    ax.add_patch(rect3)
    ax.add_patch(rect4)
#    return ax

def plot_pv(q_date, dt, mode=''):
    #dt: single ticker
    import matplotlib.pyplot as plt
    from matplotlib.finance import candlestick_ohlc as cdl
    import matplotlib.dates as mdates
    from matplotlib.dates import num2date
    import matplotlib
    import scipy.stats as stats
    ticker=dt.ticker.unique()[0]
    sig=2.5
    tick_periods=250+50
    
    dt=dt.sort_values('date', ascending=True)
    dt['ma_5']=dt['close'].rolling(5).mean()
    dt['ma_20']=dt['close'].rolling(20).mean()
    dt['ma_50']=dt['close'].rolling(50).mean()
    dt=dt[['date','open','high','low','close','volume', 'ma_5','ma_20','ma_50']]
    dt.dropna(inplace=True)
    dt=dt.tail(tick_periods)

    dt['log_rtn_d']=np.log(1+dt['close'].pct_change())
    dt['std_20']=dt['log_rtn_d'].rolling(20).std()
    dt['hv_20']=dt['std_20']*(252**0.5)
    dt['p_std_20']= dt['close']*dt['std_20']
    dt['spike_d']=(dt['close']-dt['close'].shift(1))/dt['p_std_20'].shift(1)
#norm test
    statistic, p_value = stats.shapiro(dt['log_rtn_d'])
    norm_result=p_value>0.05   
#next spike_date>3 si: copy from #ax11
#CRITICAL: as index is not continuous from tbl_pv_all for single ticker
    dt.reset_index(inplace=True)
    con_sig_d=np.abs(dt['spike_d'])>=sig
    dd=dt[con_sig_d]
    dd['idx']=dd.index
    dd['idx_diff']=dd['idx'].diff()
    d=dd['idx_diff'].describe()
    if d[0]==0:
      d[4],d[5],d[6]=0,0,0  
    dd['date']=pd.to_datetime(dd['date']).dt.date
    if not dd.empty:
        last_spike_d=dd.tail(1).date.values[0]  
    else:
        last_spike_d=datetime.datetime(2018,1,1).date()    
    d_25=last_spike_d + datetime.timedelta(int(d[4]))
    d_50=last_spike_d + datetime.timedelta(int(d[5]))
    d_75=last_spike_d + datetime.timedelta(int(d[6]))
    #date conversion for plot
    dt['date']=dt['date'].astype(str).apply(lambda x: x[:10])
    dt['date']=pd.to_datetime(dt['date'],format='%Y-%m-%d')# %H:%M:%S.%f')
    dt['date']=dt['date'].map(mdates.date2num)    
    ax_date=dt.date.values
    ax_open=dt.open.values
    ax_high=dt.high.values
    ax_low=dt.low.values
    ax_close=dt.close.values
    ax_volume=dt.volume.values
    ax_ma_5=dt.ma_5.values
    ax_ma_20=dt.ma_20.values
    ax_ma_50=dt.ma_50.values
    ax_data= zip(ax_date, ax_open, ax_high, ax_low, ax_close, ax_volume)

    if len(mode) == 0:  #default single plot
        fig, ax = plt.subplots(figsize=(8,4))
    ax=plt.gca()
    ax.plot(ax_date, ax_ma_5, 'b-', label='ma_5')
    ax.plot(ax_date, ax_ma_20, 'y-', label='ma_20')
    ax.plot(ax_date, ax_ma_50, 'g-', label='ma_50')
    plt.legend(loc="center left")
    plt.title("plot_pv: %s, norm?:%s, last_spk >%s_std: %s \nspk: %s -  %s - %s"%\
              (ticker, norm_result, sig, last_spike_d, d_25, d_50, d_75))
    cdl(ax, ax_data, width=0.6, colorup='g', colordown='r')
    if len(mode) == 0:  
    #default to plot_pv only
        # shift y-limits of the candlestick plot so that there is space at the bottom for the volume bar chart
        pad = 0.5
        yl = ax.get_ylim()
        ax.set_ylim(yl[0]-(yl[1]-yl[0])*pad,yl[1])
        ax_v = ax.twinx()
        # set the position of ax2 so that it is short (y2=0.32) but otherwise the same size as ax
        ax_v.set_position(matplotlib.transforms.Bbox([[0.125,0.1],[0.9,0.32]]))
        # make bar plots and color differently depending on up/down for the day
        pos = ax_open-ax_close <0
        neg = ax_open-ax_close >0
        ax_v.bar(ax_date[pos], ax_volume[pos],color='green',width=0.6,align='center')
        ax_v.bar(ax_date[neg], ax_volume[neg],color='red',width=0.6,align='center')
    #    plt.setp(ax.get_xticklabels(),visible=False)
    ax.set_xlim(min(ax_date),max(ax_date))
    xt = ax.get_xticks()
    new_xticks = [datetime.date.isoformat(num2date(d)) for d in xt]
    ax.set_xticklabels(new_xticks,rotation=45, horizontalalignment='right')
    #earn_dt
    x_earn_dt= datetime.datetime.strptime('2018-11-27', '%Y-%m-%d')  
    y_earn_dt=ax_close[-4]
    ax.annotate('earn_dt: %s'%x_earn_dt, xy=(x_earn_dt, y_earn_dt), xytext=(15, 15), \
                textcoords='offset points', color ='b') #, arrowprops=dict(arrowstyle='-|>'))
    plt.axvline(x_earn_dt)
    # div_dt
    x_div_dt= datetime.datetime.strptime('2018-12-10', '%Y-%m-%d') 
    y_div_dt=ax_close.min()
    ax.annotate('div_dt: %s'%x_div_dt, xy=(x_div_dt, y_div_dt), xytext=(15, 15), \
                textcoords='offset points', color='b') #, arrowprops=dict(arrowstyle='-|>'))
    # spike_dt>3sig
#    x_spk_dt= datetime.datetime.strptime(d_25, '%Y-%m-%d') 
    x_spk_dt=d_25
    y_spk_dt=ax_close[-1]
    ax.annotate('spike_dt: %s'%x_spk_dt, xy=(x_spk_dt, y_spk_dt), xytext=(15, 15), \
                textcoords='offset points', color='r') #, arrowprops=dict(arrowstyle='-|>'))
    plt.axvline(x_spk_dt)
    
    plt.ion()
    print(last_spike_d, d_25, d_50, d_75)
#    plt.show()    
#    ax.clear()

def plot_pvs(q_date, ticks='', do=''): #ticks =['x','y'], do(unop)
    ticks = [x.upper() for x in ticks]
    p_date=q_date-datetime.timedelta(350)
    qry="SELECT * FROM tbl_pv_all  wHERE date BETWEEN '%s' AND '%s'" %(p_date, q_date)
    dp=read_sql(qry)
#    dp=dp[dp.ticker.isin(ticks)]
    pd.set_option('display.expand_frame_repr', False)
    for t in ticks:
        if t in dp.ticker.unique():
            dt=dp[dp.ticker == t]
            plot_pv(q_date, dt)
    pd.set_option('display.expand_frame_repr', True)

def p_mdate(ax):
    from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MonthLocator, MONDAY, FRIDAY
    import matplotlib.ticker as mticker
    import matplotlib.dates as mdates
#    ax.xaxis.set_major_locator(mticker.MaxNLocator(60))
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_minor_locator(WeekdayLocator(MONDAY))
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax.xaxis.set_minor_formatter(DateFormatter('%d'))
    ax.fmt_xdata = DateFormatter('%d')
#    ax.set_xticks(ndays[::freq])
#    ax.set_xticklabels(date_strings[::freq], rotation=45, ha='right')
#    ax.set_xlim(ndays.min(), ndays.max())
    return ax    

def plot_base(q_date, ticks='', do=''): #ticks =['x','y'], do(unop)
#original plot_base
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
    dp=read_sql(qry)
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

def cdl_weekday(cdl_data, fmt='%b %d', freq=7, **kwargs):
        """ Wrapper that artificially spaces data to avoid gaps from weekends """
#        from matplotlib.finance import candlestick_ohlc as cdl
        import matplotlib.dates as mdates
        # Convert data to numpy array
        cdl_data_arr = np.array(cdl_data)
        cdl_data_arr2 = np.hstack(\
            [np.arange(cdl_data_arr[:,0].size)[:,np.newaxis], cdl_data_arr[:,1:]])
        ndays = cdl_data_arr2[:,0]  # array([0, 1, 2, ... n-2, n-1, n])
        
        # Convert matplotlib date numbers to strings based on `fmt`
        dates = mdates.num2date(cdl_data_arr[:,0])
        date_strings = []
        for date in dates:
            date_strings.append(date.strftime(fmt))
        # Plot candlestick chart
#        cdl(ax, ohlc_data_arr2, **kwargs)
        # Format x axis
        ax.set_xticks(ndays[::freq])
        ax.set_xticklabels(date_strings[::freq], rotation=45, ha='right')
        ax.set_xlim(ndays.min(), ndays.max())
        return cdl_data
# better sample:   https://stackoverflow.com/questions/13128647/matplotlib-finance-volume-overlay
 #   https://stackoverflow.com/questions/50203612/how-to-add-the-volume-bar-charts-in-python

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

def senti():
    #bc
    dte_min=20
    p_min=10
    pct_c_min=70
    v_opt_min=1000
    
    con_dte_bc=dbc['dte']> dte_min
    con_p_bc=dbc['price']>p_min
    
    dbc['prem']=dbc['last']* dbc['vol']
    #Filter for conviction
    dbc=dbc[con_dte & con_p]
    con_bc=(dbc['type'].str.upper()=='CALL') & (dbc['bs']=='b')
    con_sc=(dbc['type'].str.upper()=='CALL') & (dbc['bs']=='s')    
    con_bp=(dbc['type'].str.upper()=='PUT') & (dbc['bs']=='b')
    con_sp=(dbc['type'].str.upper()=='PUT') & (dbc['bs']=='s')   
    prem_bc=dbc[con_bc].prem.sum()
    prem_sc=dbc[con_sc].prem.sum()    
    prem_bp=dbc[con_bp].prem.sum()
    prem_sp=dbc[con_sp].prem.sum()       
    pr_cp= (prem_bc+ prem_sc)/(prem_bp + prem_sp)
    prem_bcsp=(prem_bc+ prem_sp)
    prem_scbp=(prem_sc+ prem_bp)  
    pr_bubr=prem_bcsp/prem_scbp
    
    #mc
    con_p_mc=dmc['p']> p_min
    con_v_opt=dmc['v_opt']> v_opt_min
    dmc=dmc[con_p_mc & con_v_opt]










































