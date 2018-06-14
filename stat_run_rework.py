# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 13:15:03 2018

@author: qli1
"""

def stat_run_rework(q_date, underlying, env='prod'): 
#simplified topdown stat 
    
    p_date=q_date-datetime.timedelta(380)
    ds_etf=stat_topdown(q_date,'etf')
    ds_sp=stat_topdown(q_date,'sp500') 
    ds_etf.rename(columns={'ticker': 'sec', 'rtn_22_pct': 'srtn_22_pct', 'rtn_66_pct':'srtn_66_pct'}, inplace=True)
    df=ds_etf.merge(ds_sp, on='sec')  #form a big table to either topdown or botom up
    
    #select top and botom 3 sector and tickers
#pivot table http://pbpython.com/pandas-pivot-table-explained.html    
    
    
    
    list_sec=['SPY','XLY','XLE','XLF','XLV','XLI','XLB','XLK','XLU','XLP','XRT', 'XHB']
    list_macro=['SPY','EEM', 'GLD','JJC', 'USO','WYDE','HYG', 'JNK', 'TLT',\
        'UUP', 'DBC', 'FLAT', 'VXX', 'SPLV', 'PBP', 'SPHB', 'EMB', 'FXI', 'GDX']
     
    list_ccy=['UUP', 'FXY', 'FXE', 'FXC', 'FXA', 'GLD']
# ranking list only
    if underlying=='sec':    
        df_stat=df_stat.loc[list_sec]
    elif underlying =='macro':
        df_stat=df_stat.loc[list_macro]
    elif underlying=='ccy':
        df_stat=df_stat.loc[list_ccy]
    else:
        pass
    
    df_stat=stat_rank(df_stat)  
    df_stat=stat_trend_new_test(df_stat)
    df_stat=stat_stage_new(df_stat, underlying, env)
    try:
        df_stat=df_stat.drop('^VIX', axis=0)
    except:
        pass
#    con_price=(df_stat.close_qdate>20) & (df_stat.close_qdate<250)
#    df_stat=df_stat[con_price]
        #overwrite tbl_stat_hist
        
#    df_stat.sort_index(inplace=Ture)
    if env=='prod' and underlying=='sp500':
        to_sql_replace(df_stat, "tbl_stat")
    elif env=='prod' and underlying=='etf':
#        df_stat=df_stat.drop('^VIX', axis=0)
        to_sql_replace(df_stat, "tbl_stat_etf")    
        to_sql_append(df_stat, "tbl_stat_etf_hist")
    elif env=='test' and underlying=='sp500':
        to_sql_replace(df_stat, "tbl_stat_test")
    elif env=='test' and underlying=='etf':
#        df_stat=df_stat.drop('^VIX', axis=0)
        to_sql_replace(df_stat, "tbl_stat_etf_test")

    print ("--- stat_run completed and replaced into tbl_stat_'%s'_'%s'---"%(underlying, env))
 
    return 

def stat_topdown(q_date, underlying):
     p_date=q_date-datetime.timedelta(380)
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
        df0['from_mean20']=df0['close_qdate']/ df0['mean_20']-1
        df0['from_mean50']=df0['close_qdate']/ df0['mean_50']-1     
        df0['from_mean200']=df0['close_qdate']/ df0['mean_200']-1
        df_st=df_st.append(df0)
   
    if underlying=='sp500':
        df_st=stat_beta(df_st)  #get sector
    elif underlying=='etf':
        secs=['XLI','XLY','XLK','XLV','XLP','XLU','XLF','XLB','XLE']  #defined in consituents list
        df_st=df_st.loc[st_st['tikcer'].isin(secs)]
#    df_stat=df_st.set_index('ticker') #ticker is set to be index
    else:
        pass
    df_stat['date']=q_date     
    df['rtn_5_pct']=df['rtn_5'].rank(pct=True)    
    df['rtn_22_pct']=df['rtn_22'].rank(pct=True)
    df['rtn_66_pct']=df['rtn_66'].rank(pct=True)



        