# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 11:18:59 2018

@author: qli1
"""

import datetime as datetime
from P_commons import read_sql, to_sql_append, to_sql_replace
import pandas as pd
#from P_stat import stat_beta
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import sqlite3 as db
import pandas.io.sql as pd_sql

DF_sp500=pd.read_csv('C:\\Users\qli1\BNS_wspace\ppy\constituents.csv')
df_etf=pd.read_csv('C:\\Users\qli1\BNS_wspace\ppy\etf.csv')

def to_sql_append(df, tbl_name):
    conn=db.connect(db)
    pd_sql.to_sql(df, tbl_name, conn, if_exists='append')

def to_sql_replace(df, tbl_name):
    conn=db.connect(db)
    pd_sql.to_sql(df, tbl_name, conn, if_exists='replace')  
    
#tbl_name is defined in query    
def read_sql(query, q_date):
    conn=db.connect('C:\\Users\qli1\BNS_wspace\ppy\db_op_20170907.db')
    df=pd_sql.read_sql(query, conn)
    return df

def stat_beta(df):
#add beta 
    DF_sp500=pd.read_csv('C:\\Users\qli1\BNS_wspace\ppy\constituents.csv')
    df_etf=pd.read_csv('C:\\Users\qli1\BNS_wspace\ppy\etf.csv')
    d0=pd.DataFrame()
    d0[['ticker','beta', 'sec']]=DF_sp500[['SYMBOL', 'beta','ETF']]
    
#    df['beta']=DF_sp500[DF_sp500['SYMBOL']==df['ticker']].beta.values[0]
    df=df.merge(d0, on='ticker', how='left')
    
    return df

def stat_run_rework(q_date, ticker='sp500'): 
#can also run ad hoc ticker eg. stat_run_rework(q_date,'fl')
    
    p_date=q_date-datetime.timedelta(380)
    ds_etf=stat_topdown(q_date,'etf')
    ds_sp=stat_topdown(q_date,'sp500') 
    
    ds_etf.rename(columns={'ticker': 'sec', 'rtn_22_pct': 'srtn_22_pct', \
            'rtn_66_pct':'srtn_66_pct', 'rtn_22':'srtn_22'}, inplace=True)
#    df=ds_etf.merge(ds_sp, on='sec')  #form a big table to either topdown or botom up
    
    show_etf=['sec', 'srtn_22_pct', 'chg_22_66', 'sartn_22','srtn_22']
    show_sp= ['ticker','rtn_22_pct','rtn_66_pct', 'rtn_22'] 

    df=ds_etf[show_etf].merge(ds_sp, on='sec') 
    
    df.sort_values(['srtn_22_pct','rtn_22_pct'],ascending=[False,False], inplace=True)
    lead=df[df.srtn_22>0].groupby(show_etf).head(3)
    lag=df[df.srtn_22<=0].groupby(show_etf).tail(3)


    if ticker !='sp500':  #if ad-hoc ticker
        df=df[df.ticker ==ticker.upper()]  #single ticker ad_hoc
        return df
        exit
    else:
        df=pd.concat([lead,lag],axis=0)
#    show=['sec', 'srtn_22_pct', 'srtn_66_pct', 'srtn_22','sartn_22',\
#           'ticker','rtn_22_pct','rtn_66_pct']   
    show_etf.extend(show_sp)
    show=show_etf
    return df[show]
    #select top and botom 3 sector and tickers
#pivot table http://pbpython.com/pandas-pivot-table-explained.html    
    
    
#    
#    list_sec=['SPY','XLY','XLE','XLF','XLV','XLI','XLB','XLK','XLU','XLP','XRT', 'XHB']
#    list_macro=['SPY','EEM', 'GLD','JJC', 'USO','WYDE','HYG', 'JNK', 'TLT',\
#        'UUP', 'DBC', 'FLAT', 'VXX', 'SPLV', 'PBP', 'SPHB', 'EMB', 'FXI', 'GDX']
#     
#    list_ccy=['UUP', 'FXY', 'FXE', 'FXC', 'FXA', 'GLD']
## ranking list only
#    if underlying=='sec':    
#        df_stat=df_stat.loc[list_sec]
#    elif underlying =='macro':
#        df_stat=df_stat.loc[list_macro]
#    elif underlying=='ccy':
#        df_stat=df_stat.loc[list_ccy]
#    else:
#        pass
#    
#    df_stat=stat_rank(df_stat)  
#    df_stat=stat_trend_new_test(df_stat)
#    df_stat=stat_stage_new(df_stat, underlying, env)
#    try:
#        df_stat=df_stat.drop('^VIX', axis=0)
#    except:
#        pass
##    con_price=(df_stat.close_qdate>20) & (df_stat.close_qdate<250)
##    df_stat=df_stat[con_price]
#        #overwrite tbl_stat_hist
#        
##    df_stat.sort_index(inplace=Ture)
#    if env=='prod' and underlying=='sp500':
#        to_sql_replace(df_stat, "tbl_stat")
#    elif env=='prod' and underlying=='etf':
##        df_stat=df_stat.drop('^VIX', axis=0)
#        to_sql_replace(df_stat, "tbl_stat_etf")    
#        to_sql_append(df_stat, "tbl_stat_etf_hist")
#    elif env=='test' and underlying=='sp500':
#        to_sql_replace(df_stat, "tbl_stat_test")
#    elif env=='test' and underlying=='etf':
##        df_stat=df_stat.drop('^VIX', axis=0)
#        to_sql_replace(df_stat, "tbl_stat_etf_test")
#
#    print ("--- stat_run completed and replaced into tbl_stat_'%s'_'%s'---"%(underlying, env))
# 
#    return 

def stat_topdown(q_date, underlying='sp500'):
    p_date=q_date-datetime.timedelta(380)
    df_st=pd.DataFrame()
    #pick up price in the range only
    if underlying=='sp500':
        query="SELECT * FROM '%s' wHERE date BETWEEN '%s' AND '%s'" %("tbl_price", p_date, q_date)
    elif (underlying=='etf')|(underlying=='sec')|(underlying=='macro')|(underlying=='ccy'):
        query="SELECT * FROM '%s' wHERE date BETWEEN '%s' AND '%s'" %("tbl_price_etf", p_date, q_date)
    else:
        print("stat_run missing underlying")
        exit
        
    df=read_sql(query, q_date)
    df.sort_index(axis=0) #sort the price from old date to latest date
    df=df.fillna(0)
    df.set_index('date', inplace=True) 
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
        df0['close_qdate']=df.iloc[-1,c]  #last row is latest price
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
#        x=df.iloc[-15:x,c]
        df_st=df_st.append(df0)
   
    df_st.drop(['25%', '50%', '75%','count','max','min','std','mean'], axis=1, inplace=True)
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
    else:
        pass
    df_st['date']=q_date     
    df_st['fm_mean20']=(df_st['close_qdate']/ df_st['mean_20'])-1
    df_st['fm_mean50']=df_st['close_qdate']/ df_st['mean_50']-1     
    df_st['fm_mean200']=df_st['close_qdate']/ df_st['mean_200']-1
    df_st['fm_hi252']=df_st['close_qdate']/ df_st['hi_252']-1
    df_st['fm_lo252']=df_st['close_qdate']/ df_st['lo_252']-1
    
    df_st['rtn_5_pct']=df_st['rtn_5'].rank(pct=True)*10
    df_st['rtn_22_pct']=df_st['rtn_22'].rank(pct=True)*10
    df_st['rtn_66_pct']=df_st['rtn_66'].rank(pct=True)*10
    
#    df_st['p_22_sig']=df_st['close_qdate']*df_st['hv_252']*np.sqrt(22)/ np.sqrt(252)
#    df_st['p_66_sig']=df_st['close_qdate']*df_st['hv_252']*np.sqrt(66)/ np.sqrt(252)
    df_st['p_22_sig']=df_st['close_qdate']*df_st['hv_22']
    df_st['p_66_sig']=df_st['close_qdate']*df_st['hv_66']
    
    return df_st

def play_candy(q_date, ticker=''): # generate play candidately by ad-hoc
    from P_intel import get_earning_ec, get_rsi
    df=stat_run_rework(q_date,ticker)
    stat_schema=df.columns
    df.reset_index(inplace=True, drop=True)
    # enrich with si
    for index, row in df.iterrows():
        df.loc[index,'earn_date']=get_earning_ec(ticker)[0]
        df.loc[index,'ex_div']=get_div(ticker)
        df.loc[index,'rsi']=get_rsi(row['ticker'])
        df.loc[index,'iv'], df.loc[index,'iv_pctl'],df.loc[index,'hv_pctl'],\
            df.loc[index,'ivol_pct']=get_opt(ticker)
    
    play_schema=['earn_date','ex_div','rsi','iv', 'ivol_pct']
    entry_schema=['act','entry_price','entry_date','estike_1','estrike_2',\
        'contracts','erisk','exit_target','expire_date','be_up',\
        'be_down','exist_pct','alert_exit','alert_stop'\
        ,'exit_date','exit_price','epnl_pct','epnl', 'days_left'\
        ,'con_1', 'con_2','con_p1','con_p2','con_ex1','con_ex2',\
        'con_ex_p1','con_ex_p2','comm','delta','play']
#    tbl_trade_schema=play_schema + entry_schema
    
#    db_view=pd.DataFrame(columns=['act','entry_price','entry_date','estike_1'\
#        ,'estrike_2','contracts','erisk','exit_target','expire_date','be_up'\
#        ,'be_down','exist_pct','alert_exit','alert_stop','exit_date'\
#        ,'exit_price','epnl_pct','epnl', 'days_left','con_1', 'con_2'\
#        ,'con_p1','con_p2','con_ex1','con_ex2','con_ex_p1','con_ex_p2'\
#        ,'comm','delta','play'])

    play_view=['ticker','beta',
       'close_qdate', 'fm_mean20', 'fm_mean50', 'fm_mean200','fm_hi252',\
       'fm_lo252','hv_22', 'hv_66',  'mean_20','mean_50', 'mean_200', 
       'sec', 'srtn_22_pct',  'chg_22_66', 'rtn_5_pct', 'rtn_22_pct', 'rtn_66_pct',\
       'sartn_22', 'srtn_22', 'rsi', 'ex_div', 'earn_date'\
       ,'iv','iv_pctl','hv_pctl','ivol_pct']
    
    entry_view= ['act','entry_price','entry_date','estike_1','estrike_2',\
        'exit_target','expire_date','be_up',\
        'be_down','con_1', 'con_2','con_p1','con_p2','comm','delta','play']  
    
    entry_view.extend(play_view)
   
    df_tbl_trade=df.append(pd.DataFrame(columns=entry_schema))  #play+entry schema (allinclusive)
#pre-fill
    df_tbl_trade['exit_date']='N'
    df_tbl_trade['expire_date']=datetime.date(2017,1,1)
    df_tbl_trade['ex_div']=datetime.date(2017,1,1)
    df_tbl_trade['estike_1']=1
    df_tbl_trade['con_p1']=100    
       
    df_play=df_tbl_trade[play_view]  #to show
    df_entry=df_tbl_trade[entry_view]  #to entry.csv
    #db_tbl_trade to save to db

    return df_play


def get_RSI(series, period):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    rs = pd.stats.moments.ewma(u, com=period-1, adjust=False) / \
         pd.stats.moments.ewma(d, com=period-1, adjust=False)
 #https://stackoverflow.com/questions/20526414/relative-strength-index-in-python-pandas   

#    data = pd.Series( [ 44.34, 44.09, 44.15, 43.61,
#                    44.33, 44.83, 45.10, 45.42,
#                    45.84, 46.08, 45.89, 46.03,
#                    45.61, 46.28, 46.28, 46.00,
#                    46.03, 46.41, 46.22, 45.64 ] )
    return 100 - 100 / (1 + rs)   