
# -*- coding: utf-8 -*-

import datetime as datetime
from P_commons import read_sql, to_sql_append, to_sql_replace
import pandas as pd
from T_intel import get_RSI
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def stat_VIEW(q_date, tick='topdown'): 
# link lead/lag tickers from sp to etf 
#"topdown" show sector abs rtn and rel rtn vs mkt. other sectors
#can also run ad hoc ticker incl. etf eg. stat_run_rework(q_date,['fl','xli'])

    qry_sp="SELECT * FROM tb_stat"
    qry_etf="SELECT * FROM tb_stat_etf"
    ds_etf=read_sql(qry_etf, q_date)
    ds_sp=read_sql(qry_sp,q_date)
    
    ds_etf_orig=ds_etf  # preserve columns the same as ds_sp
    ds_etf.rename(columns={'ticker': 'sec', 'rtn_22_pct': 'srtn_22_pct', \
            'rtn_66_pct':'srtn_66_pct', 'rtn_22':'srtn_22'}, inplace=True)
    
    show_etf=['sec', 'srtn_22_pct', 'chg_22_66', 'sartn_22','srtn_22']
    show_sp= ['ticker','rtn_22_pct','rtn_66_pct', 'rtn_22'] 

    df=ds_etf[show_etf].merge(ds_sp, on='sec') 
    
    df.sort_values(['srtn_22_pct','rtn_22_pct'],ascending=[False,False], inplace=True)
    lead=df[df.srtn_22>0].groupby(show_etf).head(3)
    lag=df[df.srtn_22<=0].groupby(show_etf).tail(3)

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
            df=df.append(df_nonsp)
            return df
            exit   
    else:
        df=pd.concat([lead,lag],axis=0)
    show_etf.extend(show_sp)
    show=show_etf
    df=df[show]


    pd.set_option('display.expand_frame_repr', False)
    print(df)
    pd.set_option('display.expand_frame_repr', True)

def stat_run_base(q_date, underlying="sp500", env="prod"):
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
    df_st['p_22_sig']=df_st['close_qdate']*df_st['hv_22']
    df_st['p_66_sig']=df_st['close_qdate']*df_st['hv_66']
    
    
    if env=='prod' and underlying=='sp500':
        to_sql_append(df_st, "tb_stat")
    elif env=='prod' and underlying=='etf':
        to_sql_append(df_st, "tb_stat_etf")    
    elif env=='test' and underlying=='sp500':
        to_sql_append(df_st, "tb_stat_test")
    elif env=='test' and underlying=='etf':
        to_sql_append(df_st, "tb_stat_etf_test")
    return df_st

def stat_PLOT(q_date):
    import matplotlib.pyplot as plt
    
    qry="SELECT * FROM tb_stat_etf"
    df=read_sql(qry, q_date)
    df.sort_values('ticker',ascending=False, inplace=True)
# secs=['XLY','XLV','XLU','XLP','XLK','XLI','XLF','XLE','XLB','SPY']  
    colors=['magenta','cyan','blue','green','red','yellow','orange'\
            'pink', 'purple','black']  

    df.plot(x=df['date'], y=df['rtn_22'], color=colors, legend=True)
    df.plot(x=df['date'], y=df['chg_22_66'], color=colors, legend=True, stacked=True, kind='bar')
    
    grid_size=[2,2]
    plt.subplot2grid(grid_size,(0,0))
    plt.plot(x=df['date'], y=df['fm_hi252'], color=colors, legend=True)
    plt.subplot2grid(grid_size,(0,1))
    plt.plot(x=df['date'], y=df['fm_mean50'], color=colors, legend=True)    
    plt.subplot2grid(grid_size,(1,0))
    plt.plot(x=df['date'], y=df['fm_lo252'], color=colors,legend=True)
    plt.subplot2grid(grid_size,(1,1))
    plt.plot(x=df['date'], y=df['fm_mean200'], color=colors, legend=True) 
    plt.show()
    
def stat_beta(df):
#add beta 
    DF_sp500=pd.read_csv('c:\\pycode\pyprod\constituents.csv')
    df_etf=pd.read_csv('c:\\pycode\pyprod\etf.csv')
    d0=pd.DataFrame()
    d0[['ticker','beta','sec']]=DF_sp500[['SYMBOL', 'beta','ETF']]
    
#    df['beta']=DF_sp500[DF_sp500['SYMBOL']==df['ticker']].beta.values[0]
    df=df.merge(d0, on='ticker', how='left')
    return df

def anytick():
    ds=stat
    tick=[e.upper() for e in tick]
    add=list(set(tick)-set(ds.ticker.tolist()))
    dadd=pd.DataFrame(columns=ds.columns)
    dadd.ticker=add
    ds.append(dadd)
    play_candy