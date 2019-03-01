
# -*- coding: utf-8 -*-

import datetime as datetime
from P_commons import read_sql, to_sql_append, to_sql_replace
import pandas as pd
from T_intel import get_RSI
import numpy as np
import warnings
warnings.filterwarnings("ignore")
pd.option_context('display.float_format', '${:,.1f}'.format)


bond=['BIL','SHV','SHY','IEF','TLT','GOVT']
eqty=['SPY','EEM','EFA','PBP','SPLV', 'SPHB']
vol=['VXX']
ccy=['UUP','FXY','FXA']
com=['JJC','DBC','USO','GLD']
cred=['WYDE', 'HYG']

def senti_mc():
    iv_chg_min=15
    pct_c_min=20
    pct_c_max=100- pct_c_min
    dm=read_sql("SELECT * FROM tbl_mc_raw")
    dm['iv_chg']=dm['iv_chg'].str.replace('%','')
    dm['iv_chg']=dm['iv_chg'].astype(float)  
    dm['p_chg']=dm['p_chg'].str.replace('%','')
    dm['p_chg']=dm['p_chg'].astype(float)
    
    dmg=dm.groupby('date')
    data=[]
    for n,g in dmg:
        data.append([n,g[g.pct_c>pct_c_max].shape[0]/g.shape[0], g[g.pct_c<pct_c_min].shape[0]/g.shape[0]])
    data=np.asarray(data)
    data=data.transpose()
    cols=['date','pct_c','pct_p']
    df=pd.DataFrame(dict(zip(cols,data)))
        
    df['pct_c']=df['pct_c'].astype(float)
    df['pct_p']=df['pct_p'].astype(float)
    df['pct_cp']=df['pct_c']/df['pct_p']
    df['c-p']=df['pct_c']-df['pct_p']
    
    dp=read_sql("SELECT * FROM tbl_pv_etf WHERE ticker='SPY'")
#    df.plot(x='date', y=['pct_c', 'pct_p'], kind='line', rot=90)
    df.plot(x='date', y=['pct_c', 'pct_p'], kind='line', rot=90)
    
    
    
def tick_leadlag(q_date, ticks=['']):
    dt=stat_VIEW(q_date, ticks)
    show=['ticker','close','abs_rtn_22', 'rtn_22_pct','beta', 'rsi',\
          'sec','srtn_22_pct']
    con_sec= pd.notnull(dt['sec'])
    print(dt[con_sec][show])
    df=dt[ ~con_sec ]
    for t in df['ticker']:
        df.loc[df.ticker==t, 'sec']=input("input sector of %s:"%t)
#        print(dt[dt.ticker==t][show])
        sec_leadlag(q_date, [sec])
    
def etf_corr():
    etf=bond+eqty+vol+ccy+com+cred
    df=read_sql("SELECT * FROM tbl_pv_all")
    df=df[df.ticker.isin(etf)]
    dp=pd.pivot_table(df, index=['date'], columns=['ticker'],values=['close'])
    dp=dp.corr()
    c_plus=dp>0.8
    c_neg=dp < -0.8
    c_no= (dp <0.1)& (dp >-0.1)
    pd.set_option('display.expand_frame_repr', False)
    print(dp[c_plus].fillna(''))
    print(dp[c_neg].fillna(''))
    print(dp[c_no].fillna(''))
        

def sec_leadlag(q_date, sectors=['ALL'], period=22, num=3): # ['XLI','XLB']
    df=read_sql("SELECT * FROM tbl_stat")
    df['date']=pd.to_datetime(df['date']).dt.date
    p_date=q_date - datetime.timedelta(period)
    sec_orig=sectors
    if sectors[0]=='ALL':
        sectors=df['sec'].unique()
    data=list()   
    for sec in sectors:
#        if sec not in df['sec'].unique():
#            print("not in sector list", sec)
#            pass
        ds=df[df.sec==sec.upper()]
        ds.sort_values(['date','rtn_22_pct'], inplace=True)
        ds=ds[ds.date>=p_date]
        lead=ds[ds.date==q_date].tail(num)
        lead['momt']='lead'
#        lead.sort_values('rtn_22_pct',ascending=False, inplace=True)
        lag=ds[ds.date==q_date].head(num)
        lag['momt']='lag'
#        lag.sort_values('rtn_22_pct',ascending=True, inplace=True)
        tag=pd.concat([lead,lag], axis=0)
        tag.sort_values('rtn_22_pct', ascending=False, inplace=True)
        for t in tag['ticker']:
            ds.loc[ds.ticker==t,'rtn_5_pct_avg']=ds[ds.ticker==t]['rtn_5_pct'].mean()
        ds_avg=ds[['ticker','rtn_5_pct_avg']]    
        ds_avg.drop_duplicates(keep='first', inplace=True)
        dm=tag.merge(ds_avg, on='ticker', how='left')
        dm.loc[dm.momt=='lead', 'agree']=dm[dm.momt=='lead']['rtn_5_pct']>dm[dm.momt=='lead']['rtn_5_pct_avg']
        dm.loc[dm.momt=='lag', 'agree']=dm[dm.momt=='lag']['rtn_5_pct']<dm[dm.momt=='lag']['rtn_5_pct_avg']
        leading=dm[(dm.momt=='lead') & (dm.agree==True)].shape[0]/num
        lagging=dm[(dm.momt=='lag') & (dm.agree==True)].shape[0]/num
        rotate='{:1f}'.format(leading-lagging)
        data.append([sec, leading, lagging, rotate])
        if sec_orig[0] !='ALL':
#            print(sec, 'leading: ', leading, "lagging: ", lagging, "rotate: ", rotate)
            show=['sec','momt','ticker','rtn_22_pct','agree', 'rtn_5_pct','rtn_5_pct_avg','rtn_22','beta']
            print (dm[show])    
    cols=['sec','leading', 'lagging', 'rotate']
    data=np.asarray(data)
    data=data.transpose()
    df_sec=pd.DataFrame(dict(zip(cols, data)))
    print(df_sec.sort_values('rotate', ascending=False))
    
def sec_stat(lookback_period=22):
    ds=read_sql("SELECT * FROM tbl_stat_etf")
    #rtn_22: relative rtn to SPY, sartn_22: abs ret
    #get correlation
    ds.sort_values('date',inplace=True)
    ds_pvt=pd.pivot_table(ds, index=['date'], columns=['ticker'], values=['sartn_22'])
    ds_corr=ds_pvt.corr()
       
    dsg=ds.groupby('ticker')
    data=[]
    for n, g in dsg:
        g.sort_values('date', inplace=True)
        g=g.tail(lookback_period)
        std_22='{:.3f}'.format(g['sartn_22'].std())
        max_22_pct='{:.0f}'.format(g['rtn_22_pct'].max())
        min_22_pct=g['rtn_22_pct'].min()
        begin_22_pct=g['rtn_22_pct'].head(1).values[0].astype(float)
        end_22_pct=g['rtn_22_pct'].tail(1).values[0].astype(float)
        sartn_22=g['sartn_22'].tail(1).values[0].astype(float)
        avg_5_pct='{:.0f}'.format(g['rtn_5_pct'].mean())
 #       ratio=(float(end_22)-float(begin_22))/std_5
        data_tmp=[n,std_22, begin_22_pct, end_22_pct, max_22_pct, \
                  min_22_pct, avg_5_pct, sartn_22]
        data.append(data_tmp)
    data=np.asarray(data)
    data=data.transpose()
    cols=['ticker','std_22','begin_22_pct','end_22_pct', 'max_22_pct',\
          'min_22_pct', 'avg_5_pct','sartn_22']
    df=pd.DataFrame(dict(zip(cols, data)))
    df['abs_sharpe']=df['sartn_22'].astype(float)/df['std_22'].astype(float)
    df['sartn_22']=df['sartn_22'].astype(float)
    df['abs_sharpe']=df['abs_sharpe'].astype(float)
    df.sort_values(['abs_sharpe', 'end_22_pct'], ascending=(False, False), inplace=True)
    show=['ticker','sartn_22','abs_sharpe', 'end_22_pct','std_22', 'avg_5_pct',\
          'begin_22_pct','max_22_pct','min_22_pct']
    
    fmt_map={'sartn_22': '{:.2f}', 'abs_sharpe': '{:.1f}'}
    for key, value in fmt_map.items():
        df[key]=df[key].apply(value.format)
    pd.set_option('display.expand_frame_repr', False)
    print(df[show])
    print("sector correlation")
    print(ds_corr.iloc[:,0].sort_values())
    print(ds_corr[ds_corr<=0.5].fillna(''))
    pd.set_option('display.expand_frame_repr', True)
    

def fmt_map(df, cols, fmt='{:.2f}'):
    df[cols]=df[cols].apply(fmt.format)
    return df
    
def test():
    
    idx_etf=dse.date.unique()
    id=np.arange(len(idx_etf)-1, 0, -5)
    s=[]
    for i in id:
        s.append(idx_etf[i])
#    idx_etf=pd.date_range(idx_etf.min(), idx_etf.max(), freq='5D', closed='right')
    dse=dse[dse.date.isin(s)]
    dsg=dse.groupby(['ticker'])
    grp=pd.DataFrame()
    for n, g in dsg:
        g.sort_values('date',ascending=True, inplace=True)
        g['delta_5']=g['rtn_5_pct']-g['rtn_5_pct'].shift(1)
        g['delta_22']=g['rtn_22_pct']-g['rtn_22_pct'].shift(1)
        g['rd_522']=g['delta_5']/g['delta_22']
        grp=pd.concat([grp, g], axis=0)
    grp.sort_values(['date', 'ticker'],inplace=True)
    grp['rd_522'].dropna(inplace=True)
    return grp


def stat_VIEW(q_date, tick='topdown'): 
#can also run ad hoc ticker incl. etf eg. stat_run_rework(q_date,['fl','xli'])
    qry_sp="SELECT * FROM tbl_stat WHERE date='%s'"%q_date
    qry_etf="SELECT * FROM tbl_stat_etf WHERE date='%s'"%q_date
    ds_etf=read_sql(qry_etf, q_date)
    ds_sp=read_sql(qry_sp, q_date)
    ds_etf_orig=ds_etf  # preserve cmns the same as ds_sp
#srtn_22_pct (sec_rtn_22 - spy_rtn_22)
    ds_etf.rename(columns={'ticker': 'sec', 'sartn_22': 'abs_rtn_22','rtn_22_pct': 'srtn_22_pct', \
            'rtn_66_pct':'srtn_66_pct', 'rtn_22':'srtn_22',\
        'fm_hi':'sfm_hi', 'fm_lo': 'sfm_lo',\
        'fm_50': 'sfm_50', 'fm_200':'sfm_200'\
        }, inplace=True)
    
#    show_etf=['sec', 'srtn_22_pct','srtn_22', 'chg_22_66', 'abs_rtn_22',\
#              'sfm_hi','sfm_lo','sfm_50','sfm_200']
#    show_sp= ['ticker','rtn_22_pct','rtn_66_pct', 'rtn_22','fm_hi', 'fm_lo',\
#              'fm_50', 'fm_200', 'close']  

    show_etf=['sec', 'srtn_22_pct', 'srtn_66_pct','abs_rtn_22','srtn_22']
    show_sp= ['ticker','rtn_22_pct','rtn_66_pct', 'rtn_22','close']      
    show=show_sp+ show_etf
    df=ds_etf[show_etf].merge(ds_sp, on='sec') 
    df.sort_values(['srtn_22'],ascending=[False], inplace=True)

#    lead=df[df.srtn_22>0].groupby(show_etf).head(3)
    lead=df[df.srtn_22>0].sort_values(['rtn_22_pct'], ascending=False).groupby(show_etf).head(3)
    lag=df[df.srtn_22<=0].sort_values(['rtn_22_pct'], ascending=False).groupby(show_etf).tail(3)
#    if tick !='topdown':  #if ad-hoc ticker
#            tick=[x.upper() for x in tick]
#            df=df[df['ticker'].isin(tick)]            
#            return df
#            exit
    if tick !='topdown':  #if ad-hoc ticker
            tick=[x.upper() for x in tick]
            con_sp=df['ticker'].isin(tick)
            ds_etf_orig=read_sql(qry_etf, q_date)
            con_etf=ds_etf_orig['ticker'].isin(tick)
            
            df_sp=df[con_sp]
            df_etf=ds_etf_orig[con_etf]      
            df=pd.concat([df_sp, df_etf], axis=0) 
#append row of tickers not in sp500 or sp500 ETF            
            tick_nonsp= list(set(tick)-set(df.ticker.tolist()))
            df_nonsp=pd.DataFrame(columns=df.columns)
            df_nonsp.ticker=tick_nonsp
#get non_sp stat only when not empty
            if not df_nonsp.empty: 
                df_nonsp=stat_run_base(q_date, 'ad_hoc', 'prod', df_nonsp)
            df=df.append(df_nonsp[show_sp])
            return df[show]
            exit   
    else:
        df=pd.concat([lead,lag],axis=0)
#    show_etf.extend(show_sp)
#    show=show_etf
    df=df[show]
    df.sort_values(['srtn_22','rtn_22_pct'],ascending=[False, False], inplace=True)

    pd.set_option('display.expand_frame_repr', False)
    print(df)
    pd.set_option('display.expand_frame_repr', True)
    return df
    

#
def stat_run_base(q_date, underlying='sp500', env='prod',df_ah=''):
    '''
    In:tbl_price (since 2013), tbl_pv (since 2018 jun)
    Update: tbl_stat, tbl_stat_etf
    note:
    tbl_stat_ep, etf is created for quicker access
    ticker rtn_22_pct in 500 component, sec rtn_22_pct in 8 etfs
    '''
    p_date=q_date-datetime.timedelta(380)
    df_st=pd.DataFrame()
    #pick up price in the range only
#    if underlying=='sp500':
#        query="SELECT * FROM '%s' wHERE date BETWEEN '%s' AND '%s'" %("tbl_price", p_date, q_date)
#        df=read_sql(query, q_date)
#    elif (underlying=='etf')|(underlying=='sec')|(underlying=='macro')|(underlying=='ccy'):
#        query="SELECT * FROM '%s' wHERE date BETWEEN '%s' AND '%s'" %("tbl_price_etf", p_date, q_date)
#        df=read_sql(query, q_date)
    if underlying=='sp500':
        query="SELECT * FROM '%s' wHERE date BETWEEN '%s' AND '%s'" %("tbl_pv_sp500", p_date, q_date)
        df=read_sql(query, q_date)
    elif (underlying=='etf'):
        query="SELECT * FROM '%s' wHERE date BETWEEN '%s' AND '%s'" %("tbl_pv_etf", p_date, q_date)
        df=read_sql(query, q_date)        
    elif (underlying=='ad_hoc') and  (~ df_ah.empty):
        qry="SELECT * FROM '%s' wHERE date BETWEEN '%s' AND '%s'" %("tbl_pv_all", p_date, q_date)
        df=read_sql(qry, q_date)
        df=df[df.ticker.isin(df_ah.ticker)]

    else:
        print("stat_run missing underlying")
        exit

    df_pivot=df.pivot_table(index='date', columns='ticker', values='close')
    df=pd.DataFrame(df_pivot.to_records())
     #sort the price from old date to latest date

    df.set_index('date', inplace=True) 
    df.sort_index(axis=0, ascending=True, inplace=True)
    if df.isnull().values.any():
        df.fillna(method='ffill',inplace=True)
    df0=pd.DataFrame()
    len=df.shape[1]  
    #iterate thru columns, each column is a ticker
    #if iterate thru rows then use below code
# for index, row in df.iterrows():
#   ....:     print row['c1'], row['c2']   
    #for columns in df (another way)
#close_qdate, mean_20,50,200, hi_252,lo_252, 
    for c in range (0,len):
        df0=df.iloc[:,c].describe()
        #df0['ticker']=df.iloc[0:,c].name
        df0['ticker']=df.columns[c]
        df0['close']=df.iloc[-1,c]  #last row is latest price
        df0['close_22b']=df.iloc[-22,c]
        df0['close_66b']=df.iloc[-66,c]
        df0['mean_20']=df.iloc[:,c].tail(20).mean()
        df0['mean_50']=df.iloc[:,c].tail(50).mean()
        df0['mean_200']=df.iloc[:,c].tail(200).mean()
        df0['hi_252']=df.iloc[:,c].tail(252).max()  
        df0['lo_252']=df.iloc[:,c].tail(252).min()  
        df0['rtn_5']=df.iloc[-1,c]/df.iloc[-5,c]-1 
        df0['rtn_22']=df.iloc[-1,c]/df.iloc[-22,c]-1 
        df0['rtn_66']=df.iloc[-1,c]/df.iloc[-66,c]-1 
        log_rtn=np.log(df.iloc[:,c]/df.iloc[:,c].shift(1))
        df0['hv_22']=np.sqrt(252*log_rtn.tail(22).var())
        df0['hv_66']=np.sqrt(252*log_rtn.tail(66).var())
        df0['rsi']=get_RSI(pd.Series(df.iloc[-15:,c].values),14).values[0]
        df_st=df_st.append(df0)
    try:
        df_st.drop(['25%', '50%', '75%','count','max','min','std','mean'], axis=1, inplace=True)
    except:
        pass
   
    if underlying=='sp500':
        df_st=stat_beta(df_st)  #get sector
        df_st.dropna(inplace=True)
    elif underlying=='etf':
        secs=['XLI','XLY','XLK','XLV','XLP','XLU','XLF','XLB','XLE','SPY']  #defined in consituents list
        df_st=df_st.loc[df_st['ticker'].isin(secs)]
        df_st['sartn_22']=df_st['rtn_22']
        df_st['chg_22_66']=df_st['rtn_22']-df_st['rtn_66']
        spy_rtn_22=df_st[df_st.ticker=='SPY']['rtn_22'].values[0]
        spy_rtn_66=df_st[df_st.ticker=='SPY']['rtn_66'].values[0]
        df_st['rtn_22']-= spy_rtn_22
        df_st['rtn_66']-= spy_rtn_66
#    df_stat=df_st.set_index('ticker') #ticker is set to be index
    elif underlying=='ad_hoc':
        df_st_orig=df_st.sort_values('ticker')  #reserve original df_st
        df_sp=read_sql("SELECT * FROM tbl_stat")
        df_sp.drop('index',axis=1, inplace=True)
        df_sp=df_sp[df_sp.date==df_sp.date.max()]
        #merge df_st with df_sp to get ad_hoc ticker rank with sp components
        df_st=pd.concat([df_sp, df_st], axis=0)
    else:
        pass
    df_st['date']=q_date     
    df_st['fm_mean20']=(df_st['close']/ df_st['mean_20'])-1
    df_st['fm_50']=df_st['close']/ df_st['mean_50']-1     
    df_st['fm_200']=df_st['close']/ df_st['mean_200']-1
    df_st['fm_hi']=df_st['close']/ df_st['hi_252']-1
    df_st['fm_lo']=df_st['close']/ df_st['lo_252']-1
    
    df_st['rtn_5_pct']=df_st['rtn_5'].rank(pct=True)*10
    df_st['rtn_22_pct']=df_st['rtn_22'].rank(pct=True)*10
    df_st['rtn_66_pct']=df_st['rtn_66'].rank(pct=True)*10
    df_st['p_22_sig']=df_st['close']*df_st['hv_22']
    df_st['p_66_sig']=df_st['close']*df_st['hv_66']
    
    if underlying=='ad_hoc': 
        #for non-sp ticker
        df_st=df_st[df_st.ticker.isin(df_st_orig.ticker)]
    
    if env=='prod' and underlying=='sp500':
        to_sql_append(df_st, "tbl_stat")
    elif env=='prod' and underlying=='etf':
        to_sql_append(df_st, "tbl_stat_etf")    
    elif env=='test' and underlying=='sp500':
        to_sql_append(df_st, "tbl_stat_test")
    elif env=='test' and underlying=='etf':
        to_sql_append(df_st, "tbl_stat_etf_test")
    return df_st

def stat_beta(df):
#add beta 
    DF_sp500=pd.read_csv('c:\\pycode\pyprod\constituents.csv')
    df_etf=pd.read_csv('c:\\pycode\pyprod\etf.csv')
    d0=pd.DataFrame()
    d0[['ticker','beta','sec']]=DF_sp500[['SYMBOL', 'beta','ETF']]
    
#    df['beta']=DF_sp500[DF_sp500['SYMBOL']==df['ticker']].beta.values[0]
    df=df.merge(d0, on='ticker', how='left')
    return df
def stat_PLOT(q_date):
    import matplotlib.pyplot as plt
    
    qry="SELECT * FROM tbl_stat_etf"
    df=read_sql(qry, q_date)
    df.drop(['index'], axis=1, inplace=True) #
    df.sort_values('ticker',ascending=False, inplace=True)
# secs=['XLY','XLV','XLU','XLP','XLK','XLI','XLF','XLE','XLB','SPY']  
    pvt=df.pivot(index='date',columns='ticker',values='rtn_22')
#    colors=['magenta','cyan','blue','green','red','yellow','orange',\
#            'pink', 'purple','black']  
    colors=['black','purple','pink', 'orange','yellow',\
            'red','green','blue','cyan','magenta']

    pvt.plot(kind='line', color=colors, figsize=(16,8), rot=45)
    plt.legend(fontsize=15, loc='best')

    plt.show()
    
def anytick():
    ds=stat
    tick=[e.upper() for e in tick]
    add=list(set(tick)-set(ds.ticker.tolist()))
    dadd=pd.DataFrame(columns=ds.columns)
    dadd.ticker=add
    ds.append(dadd)
    play_candy
    
def get_beta():
    df=read_sql("SELECT * FROM tbl_pv_etf")
    secs=['XLI','XLY','XLK','XLV','XLP','XLU','XLF','XLB','XLE','SPY']
    df=df[df.ticker.isin(secs)]
    dfg=df.groupby('ticker')
    x=dfg.get_group('SPY')
    x.sort_values('date', inplace=True)
    x['mrtn_22']=x['close']/x['close'].shift(22)
    x.dropna(inplace=True)
    for n, g in dfg:
        g.sort_values('date',inplace=True)
        g['rtn_22']=g['close']/g['close'].shift(22)
        g.dropna(inplace=True)
        gx=pd.merge(g,x[['date','mrtn_22']], how='left',on='date')
        win=132
        gx['beta'] = pd.rolling_cov(gx['rtn_22'], gx['mrtn_22'], window=win) \
            / pd.rolling_var(gx['mrtn_22'], window=win)
        gx.dropna(inplace=True)
        gx.plot(x='date',y='beta',kind='line', title=n)

def vwap(df):
        q = df.quantity.values
        p = df.price.values
        df.assign(vwap=(p * q).cumsum() / q.cumsum())    
        df = df.groupby(df.index.date, group_keys=False).apply(vwap)  
        
def dev(q_date, underlying='sp500', env='prod',df_ah=''):
    '''
    Run stat_run_base(etf) first then (sp)
    In:tbl_price (since 2013), tbl_pv (since 2018 jun)
    Update: tbl_stat, tbl_stat_etf
    note:
    tbl_stat_ep, etf is created for quicker access
    ticker rtn_22_pct in 500 component, sec rtn_22_pct in 8 etfs
    '''
    import scipy.stats as stats
    p_date=q_date-datetime.timedelta(380)
    
    #pick up price in the range only
#    if underlying=='sp500':
#        query="SELECT * FROM '%s' wHERE date BETWEEN '%s' AND '%s'" %("tbl_price", p_date, q_date)
#        df=read_sql(query, q_date)
#    elif (underlying=='etf')|(underlying=='sec')|(underlying=='macro')|(underlying=='ccy'):
#        query="SELECT * FROM '%s' wHERE date BETWEEN '%s' AND '%s'" %("tbl_price_etf", p_date, q_date)
#        df=read_sql(query, q_date)
    if underlying=='sp500':
        query="SELECT * FROM '%s' wHERE date BETWEEN '%s' AND '%s'" %("tbl_pv_sp500", p_date, q_date)
        df=read_sql(query, q_date)
    elif (underlying=='etf'):
        query="SELECT * FROM '%s' wHERE date BETWEEN '%s' AND '%s'" %("tbl_pv_etf", p_date, q_date)
        df=read_sql(query, q_date)        
    elif (underlying=='ad_hoc') and  (~ df_ah.empty):
        qry="SELECT * FROM '%s' wHERE date BETWEEN '%s' AND '%s'" %("tbl_pv_all", p_date, q_date)
        df=read_sql(qry, q_date)
        df=df[df.ticker.isin(df_ah.ticker)]
    else:
        print("stat_run missing underlying")
        exit
    ds=pd.DataFrame()
    dfg=df.groupby('ticker')
#daily, calc stat for past 1 year from q_date as the stat of q_date
    for n,g in dfg:  #new item: std_22, std_66, spike, p_value, fm_20
        g.sort_values('date',inplace=True)
        g['close_22b']=g['close'].shift(22)
        g['close_66b']=g['close'].shift(66)
        g['mean_20']=g['close'].tail(20).mean()
        g['mean_50']=g['close'].tail(50).mean()
        g['mean_200']=g['close'].tail(200).mean()
        g['hi_252']=g['close'].tail(252).max()
        g['lo_252']=g['close'].tail(252).min()
        g['rtn_5']=g['close']/g['close'].shift(5)-1
        g['rtn_22']=g['close']/g['close'].shift(22)-1
        g['rtn_66']=g['close']/g['close'].shift(66)-1
        g['std_22']=(np.log(1+g['close'].pct_change())).rolling(22).std()
        g['hv_22']=g['std_22']*(252**0.5)
        g['std_66']=(np.log(1+g['close'].pct_change())).rolling(66).std()
        g['hv_66']=g['std_66']*(252**0.5)
#        g['rsi']=get_RSI()
        g['spike']=(g['close']-g['close'].shift(1))/(g['std_22'].shift(1)*g['close'].shift(1))
        g['p_value']=stats.shapiro(np.log(1+g['close'].pct_change()))[1]
#       g['beta']=get_BETA()
        g['fm_20']=g['close']/g['mean_20']-1
        g['fm_50']=g['close']/g['mean_50']-1
        g['fm_200']=g['close']/g['mean_200']-1
        g['fm_hi']=g['close']/g['hi_252']-1
        g['fm_lo']=g['close']/g['lo_252']-1
        g['p_22_sig']=g['close']*g['hv_22']*np.sqrt(22/252)
        g['p_66_sig']=g['close']*g['hv_66']*np.sqrt(66/252)
#        g['vwap']
#get the last row as stat as of q_date
        ds_qdate=g.tail(1)
        ds=pd.concat([ds, ds_qdate], axis=0)
        
    if underlying=='sp500':
        ds=stat_beta(ds)  #get sector
        ds.dropna(inplace=True)
    elif underlying=='etf':
    #rename columns, rtn_22 -> relative rtn to SPY
        secs=['XLI','XLY','XLK','XLV','XLP','XLU','XLF','XLB','XLE','SPY']  #defined in consituents list
        ds=ds.loc[ds['ticker'].isin(secs)]
        ds['sartn_22']=ds['rtn_22']
        ds['chg_22_66']=ds['rtn_22']-ds['rtn_66']
        spy_rtn_22=ds[ds.ticker=='SPY']['rtn_22'].values[0]
        spy_rtn_66=ds[ds.ticker=='SPY']['rtn_66'].values[0]
        ds['rtn_22']-= spy_rtn_22
        ds['rtn_66']-= spy_rtn_66
#    dsat=ds.set_index('ticker') #ticker is set to be index
    elif underlying=='ad_hoc':
        #only non_sp ticker go this route
        ds_orig=ds.sort_values('ticker')  #reserve original ds
        df_sp=read_sql("SELECT * FROM tbl_stat")
        df_sp.drop('index',axis=1, inplace=True)
        df_sp=df_sp[df_sp.date==df_sp.date.max()]
        #merge ds with df_sp to get ad_hoc ticker rank with sp components
        ds=pd.concat([df_sp, ds], axis=0)
    else:
        pass
# rank rtn_pct among sp components as of q_date    
    ds['rtn_5_pct']=ds['rtn_5'].rank(pct=True)*10
    ds['rtn_22_pct']=ds['rtn_22'].rank(pct=True)*10
    ds['rtn_66_pct']=ds['rtn_66'].rank(pct=True)*10

    if underlying=='ad_hoc': 
        #for non-sp ticker
        ds=ds[ds.ticker.isin(ds_orig.ticker)]
    
    if env=='prod' and underlying=='sp500':
        to_sql_append(ds, "tbl_stat")
    elif env=='prod' and underlying=='etf':
        to_sql_append(ds, "tbl_stat_etf")    
    return ds
#        
        
#    df_pivot=df.pivot_table(index='date', columns='ticker', values='close')
#    df=pd.DataFrame(df_pivot.to_records())
#     #sort the price from old date to latest date
#
#    df.set_index('date', inplace=True) 
#    df.sort_index(axis=0, ascending=True, inplace=True)
#    if df.isnull().values.any():
#        df.fillna(method='ffill',inplace=True)
#    df0=pd.DataFrame()
#    len=df.shape[1]  
#    #iterate thru columns, each column is a ticker
#    #if iterate thru rows then use below code
## for index, row in df.iterrows():
##   ....:     print row['c1'], row['c2']   
#    #for columns in df (another way)
##close_qdate, mean_20,50,200, hi_252,lo_252, 
#    for c in range (0,len):
#        df0=df.iloc[:,c].describe()
#        #df0['ticker']=df.iloc[0:,c].name
#        df0['ticker']=df.columns[c]
#        df0['close']=df.iloc[-1,c]  #last row is latest price
#        df0['close_22b']=df.iloc[-22,c]
#        df0['close_66b']=df.iloc[-66,c]
#        df0['mean_20']=df.iloc[:,c].tail(20).mean()
#        df0['mean_50']=df.iloc[:,c].tail(50).mean()
#        df0['mean_200']=df.iloc[:,c].tail(200).mean()
#        df0['hi_252']=df.iloc[:,c].tail(252).max()  
#        df0['lo_252']=df.iloc[:,c].tail(252).min()  
#        df0['rtn_5']=df.iloc[-1,c]/df.iloc[-5,c]-1 
#        df0['rtn_22']=df.iloc[-1,c]/df.iloc[-22,c]-1 
#        df0['rtn_66']=df.iloc[-1,c]/df.iloc[-66,c]-1 
#        log_rtn=np.log(df.iloc[:,c]/df.iloc[:,c].shift(1))
#        df0['hv_22']=np.sqrt(252*log_rtn.tail(22).var())
#        df0['hv_66']=np.sqrt(252*log_rtn.tail(66).var())
#        df0['rsi']=get_RSI(pd.Series(df.iloc[-15:,c].values),14).values[0]
#        df_st=df_st.append(df0)
#    try:
#        df_st.drop(['25%', '50%', '75%','count','max','min','std','mean'], axis=1, inplace=True)
#    except:
#        pass
#   
#    if underlying=='sp500':
#        df_st=stat_beta(df_st)  #get sector
#        df_st.dropna(inplace=True)
#    elif underlying=='etf':
#        secs=['XLI','XLY','XLK','XLV','XLP','XLU','XLF','XLB','XLE','SPY']  #defined in consituents list
#        df_st=df_st.loc[df_st['ticker'].isin(secs)]
#        df_st['sartn_22']=df_st['rtn_22']
#        df_st['chg_22_66']=df_st['rtn_22']-df_st['rtn_66']
#        spy_rtn_22=df_st[df_st.ticker=='SPY']['rtn_22'].values[0]
#        spy_rtn_66=df_st[df_st.ticker=='SPY']['rtn_66'].values[0]
#        df_st['rtn_22']-= spy_rtn_22
#        df_st['rtn_66']-= spy_rtn_66
##    df_stat=df_st.set_index('ticker') #ticker is set to be index
#    elif underlying=='ad_hoc':
#        df_st_orig=df_st.sort_values('ticker')  #reserve original df_st
#        df_sp=read_sql("SELECT * FROM tbl_stat")
#        df_sp.drop('index',axis=1, inplace=True)
#        df_sp=df_sp[df_sp.date==df_sp.date.max()]
#        #merge df_st with df_sp to get ad_hoc ticker rank with sp components
#        df_st=pd.concat([df_sp, df_st], axis=0)
#    else:
#        pass
#    df_st['date']=q_date     
#    df_st['fm_mean20']=(df_st['close']/ df_st['mean_20'])-1
#    df_st['fm_50']=df_st['close']/ df_st['mean_50']-1     
#    df_st['fm_200']=df_st['close']/ df_st['mean_200']-1
#    df_st['fm_hi']=df_st['close']/ df_st['hi_252']-1
#    df_st['fm_lo']=df_st['close']/ df_st['lo_252']-1
#    
#    df_st['rtn_5_pct']=df_st['rtn_5'].rank(pct=True)*10
#    df_st['rtn_22_pct']=df_st['rtn_22'].rank(pct=True)*10
#    df_st['rtn_66_pct']=df_st['rtn_66'].rank(pct=True)*10
#    df_st['p_22_sig']=df_st['close']*df_st['hv_22']
#    df_st['p_66_sig']=df_st['close']*df_st['hv_66']
#    
#    if underlying=='ad_hoc': 
#        #for non-sp ticker
#        df_st=df_st[df_st.ticker.isin(df_st_orig.ticker)]
#    
#    if env=='prod' and underlying=='sp500':
#        to_sql_append(df_st, "tbl_stat")
#    elif env=='prod' and underlying=='etf':
#        to_sql_append(df_st, "tbl_stat_etf")    
#    elif env=='test' and underlying=='sp500':
#        to_sql_append(df_st, "tbl_stat_test")
#    elif env=='test' and underlying=='etf':
#        to_sql_append(df_st, "tbl_stat_etf_test")
#    return df_st