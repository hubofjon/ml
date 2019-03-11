# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 10:46:42 2017
Purpose:  stat_option
Functions: get_option
@author: jon
"""
import pandas as pd
import numpy as np
import datetime as datetime
import time
from P_commons import bdate, tbl_nodupe_append, get_profile
from P_commons import to_sql_append, to_sql_replace, read_sql
pd.options.display.float_format = '{:.2f}'.format
from termcolor import colored
from timeit import default_timer as timer
from T_intel import get_earnings_ec, get_earning_ec, get_DIV, get_si_sqz, get_rsi, get_RSI
#from R_plot import plot_base
from R_plot_190206 import plot_base
from datetime import timedelta
from R_stat_dev import stat_VIEW #stat_VIEW
from dateutil import parser
from timeit import default_timer as timer
import os

def candy_combo(q_date):
    print("candy_combo started")
    start_time=timer()
# consolidate mc_candy, bc_candy, bmc_candy into one table tbl_candies in past x days
    dmct=read_sql("SELECT * FROM tbl_mc_candy", q_date)
    dbct=read_sql("SELECT * FROM tbl_bc_candy", q_date)
    dbmct=read_sql("SELECT * FROM tbl_bmc_candy", q_date)
    df=pd.concat([dmct, dbct, dbmct])
    df['date']=pd.to_datetime(df['date'])
    df=df[df.date==q_date]
    
    scm_t_bmct=dbmct.columns.tolist()
    df=df[scm_t_bmct]
    df.sort_values(['ticker','strike'], ascending=True, inplace=True)
    df.drop_duplicates(['ticker','date'], keep='first', inplace=True)
#enrich 
    if 'sec' in df.columns:
        df.drop('sec', axis=1, inplace=True)
    df=get_profile(df)
    scm_t_pro=['ticker', 'mkt', 'beta', 'sec']
    ds=stat_VIEW(q_date, df['ticker'].tolist())
    scm_v_stat=ds.columns.tolist()
    df=pd.merge(df, ds, how='left',on=scm_t_pro)

#extr step to convert iv to float 
    df['iv']=df['iv'].astype('str')
    df['iv']=df['iv'].str.replace("%",'')
    df['iv']=df['iv'].astype('float')
    df['p']=df['p'].astype('float')
#default sig period is 22
    dte=22  
    df['sig_dte']=df['p']*df['iv']/100*np.sqrt(dte/252)
    df['iv_hv']=df['iv']*0.01/df['hv_22']
    
    for index, row in df.iterrows():
        try:
            df.loc[index,'earn_dt']=get_earning_ec(row.ticker)
        except:
            pass
        try:
            df.loc[index,'div_dt']=get_DIV(row.ticker)[0]
        except:
            pass
        try:
            df.loc[index, 'si']=get_si_sqz(row.ticker)[0]
        except:
            df.loc[index, 'si']=0
        try:
            df.loc[index, 'rsi']=int(float(get_rsi(row.ticker)))
        except:
            df.loc[index, 'rsi']=0
            pass    
    
    col_extra1=['earn_dt','div_dt','si', 'rsi']
    col_extra=['sig_dte', 'iv_hv']
    scm_t_candy=scm_t_bmct + scm_t_pro + scm_v_stat + col_extra+col_extra1
    #duplicate list
    scm_t_candy=list(set(scm_t_candy))
    
    scm_t_candy=[x for x in scm_t_candy if x not in ['index','level_0']]
    df=df[scm_t_candy]
    tbl_nodupe_append(q_date, df, 'tbl_candies')

    pd.set_option('display.expand_frame_repr', False)
    df.sort_values(['v_oi','v_pct'], ascending=[False, False], inplace=True)
    df['earn_dt']=pd.to_datetime(df['earn_dt']).dt.strftime('%m/%d')
    df['event_dt']=pd.to_datetime(df['event_dt']).dt.strftime('%m/%d')
#    df.fillna('', inplace=True)
    show_candy=['ticker','sec','type','bs', 'dte','spike', 'rtn_22_pct', 'iv_hv', \
                'iv','iv_chg','beta', 'close','strike','oexp_dt',\
                'earn_dt','div_dt','event_dt','sig_dte',\
                'p_chg','v_stk_chg','fm_50']
    pd.options.display.float_format = '{:.1f}'.format
    print(df[show_candy].to_string(index=False))
    pd.set_option('display.expand_frame_repr', True)
    end_time=timer()
    print("time: ", end_time-start_time)
    pd.options.display.float_format = '{:.2f}'.format
    return df


#Source:get_unop_mc, Dest: tbl_mc_raw, no enrich, no depednet on pv_all
def unop_mc(q_date): #
    print(" ------ chamerlain unop_mc starts ------")
    fm_last_earn_dt=60
    pages, mc_date=get_unop_mc()
    if mc_date !=q_date:
        print("mc_date not equal q_date")
        return
    pages=[x for x in pages if x]  #remove empty element

    df=pd.DataFrame(data=pages[1:], columns=pages[0], index=range(len(pages)-1))
    df.rename(columns={'Symbol':'ticker', 'Volume':'v_opt', 'Calls':'pct_c', \
           'Puts':'pct_p', 'Relative Volume':'v_pct', 'Price':'p',\
           '% Chg':'p_chg', 'Earnings':'earn_dt', 'Event':'event',\
           'IV':'iv', 'IV Chg':'iv_chg'}, inplace=True)
    df['v_pct']=df.v_pct.astype(float)
    df['p_chg_abs']=df.p_chg.str.replace("+","").str.replace("-","")\
        .str.replace("%","").astype(float)
    df['p_chg']=df.p_chg.str.replace("+","")\
        .str.replace("%","").astype(float)        
    df['iv_chg_abs']=  df.iv_chg.str.replace("+","").str.replace("-","")\
        .str.replace("%","").astype(float)
    df['iv_chg']=  df.iv_chg.str.replace("+","")\
        .str.replace("%","").astype(float)
    df['v_opt']=df['v_opt'].str.replace(",", "").astype(float)
    df['v_opt_avg']=df['v_opt']/df['v_pct']
    def rgx(val):
        import re
        evt=re.search('(\d{1,2}-\S{3}-\d{4})',val)
        if evt:
            return evt.group(0)
        else:
            return np.nan        
    df['event_dt']=df['event'].apply(rgx)
#Filer outdated earn_dt from mc website   
#deal with datetime type issue: earn_dt<q_date, but no more than 2mth old
#    df['earn_dt'].fillna('30-Jan-2015 BMO', inplace=True)
    df['earn_dt']= pd.to_datetime(df['earn_dt'].str.split(' ').str[0])
    df['earn_dt'].fillna('2018-01-01', inplace=True)
    df['earn_dt']=pd.to_datetime(df['earn_dt'])

    try:
        con_earn_dt_1= df['earn_dt']<q_date
        con_earn_dt_2= (q_date -df['earn_dt'].dt.date)<datetime.timedelta(fm_last_earn_dt)
        con_earn_dt=con_earn_dt_1 & con_earn_dt_2 
        df.loc[con_earn_dt, 'earn_dt']=np.datetime64('NaT')    
    except:
        print(" --con_earn_dt type error --")
        pass
    df.sort_values(by=['v_pct','iv_chg_abs', 'p_chg_abs'], ascending=[False, True, True]
                ,axis=0, inplace=True)

    df['pct_c']=df['pct_c'].str.replace("%","").astype(float)
    df['date']=q_date
#datetime.datetime.strptime(mc_date, '%Y-%m-%d').date()
    df['date']=pd.to_datetime(df['date'])
#Prevent dupe entry to tbl_mc_raw    
#    date_exist=read_sql("SELECT DISTINCT date FROM tbl_mc_raw",q_date)
#    if mc_date in date_exist['date'][0]:
#        print("%s in tbl_mc_raw already"%mc_date)
#    else:
#        to_sql_append(df, 'tbl_mc_raw')
#        print("tbl_mc_raw is updated")
    tbl_nodupe_append(q_date, df, 'tbl_mc_raw')    
    return df

#Source tbl_mc_raw @q_date, stat_VIEW, Append to tbl_mc_candy
def mc_intel(q_date):
    print("--- mc_intel started  --- ")
    qry="SELECT * FROM tbl_mc_raw"
    df=read_sql(qry, q_date)
    df['date']=pd.to_datetime(df['date'])
    df=df[df.date==q_date]
    df['iv_chg']=df['iv_chg'].astype(float)
    df['p_chg']=df['p_chg'].astype(float)    
    df['pct_c']=df['pct_c'].astype(float)
    df['p']=df['p'].astype(float)
    df['v_opt_avg']=df['v_opt_avg'].astype(float)    
    df['v_pct']=df['v_pct'].astype(float)
    df['v_opt']=df['v_opt'].astype(float)
#criteria
    volume_opt_ratio=2
    volume_min=1000
    p_chg_pct_top=3
    iv_chg_pct_top=5
    call_pct=75
    put_pct=75
    liq=1000  
    
# REDUCE for less tickers
    con_p=(df.p<200) & (df.p>2)
    con_liq=(df.v_opt_avg >liq)
    con_vol_ratio=df.v_pct>volume_opt_ratio
    con_vol_min= df.v_opt>volume_min

    df=df[con_liq & con_p & con_vol_ratio & con_vol_min]    
#bc_interl criteria
    con_iv_up_top=df['iv_chg']>iv_chg_pct_top
    con_iv_up_lo= (~con_iv_up_top) & (df['iv_chg']>0)
    con_iv_dn_top=df['iv_chg']< (0- iv_chg_pct_top)
    con_iv_dn_lo= (~ con_iv_dn_top) & (df['iv_chg']<0)
                
    con_p_up_top=df['p_chg']>p_chg_pct_top    
    con_p_up_lo= (~con_p_up_top) & (df['p_chg']>0)
    con_p_dn_top=df['p_chg']< (0-p_chg_pct_top)
    con_p_dn_lo= (~con_p_dn_top) & (df['p_chg']<0)
    
    con_c=df['pct_c']>call_pct
    con_p=(100-df['pct_c'])>put_pct
                 
    con_cat=pd.notnull(df['earn_dt'])|pd.notnull(df['event_dt'])
#    con_iv_rank    

    CON_iv_up_top_p_up_top=con_iv_up_top & con_p_up_top #& (~con_cat)  #LC->SCV, BW)
    CON_iv_up_top_p_up_lo=con_iv_up_top & con_p_up_lo  & (con_c | con_p) #con_cat # !! Accumu ->LC, LCV
    CON_iv_up_top_p_dn_top=con_iv_up_top & con_p_dn_top & (con_c | con_p) #& (~con_cat) #LP -> BF, CAL
    CON_iv_up_top_p_dn_lo=con_iv_up_top & con_p_dn_lo & con_cat  # !! Acumu: LP or LC- up to iv rank
    
    CON_iv_dn_top_p_up_top_cp=con_iv_dn_top & con_p_up_top & (con_c |con_p) & (con_cat) #SC/SP-> LP wait)
    CON_iv_dn_top_p_up_lo= con_iv_dn_top & con_p_up_lo & con_c & ( con_cat) #SC-> wait to LP
    CON_iv_dn_top_p_dn_top=con_iv_dn_top & con_p_dn_top & (con_cat)  #more dn ?
    CON_iv_dn_top_p_dn_lo=con_iv_dn_top & con_p_dn_lo & (con_c |con_p) & con_cat #sc/sp->LC
    
    CON_iv_up_lo_dn_p_up_top=(con_iv_up_lo | con_iv_dn_lo) & con_p_up_top & (con_c |con_p) & (~ con_cat) #sc/sp->BW, Cal
    CON_iv_up_lo_dn_p_dn_top=(con_iv_up_lo | con_iv_dn_lo) & con_p_dn_top  & con_cat #sc/sp nohope-> l=LP, LPV, more down                                              
    

    
    df.loc[CON_iv_up_top_p_up_top,'play']='SCV, BW'
    df.loc[CON_iv_up_top_p_up_lo, 'play']='LCV'
    df.loc[CON_iv_up_top_p_dn_top, 'play']='BF/CAL'
    df.loc[CON_iv_up_top_p_dn_lo, 'play']='LCP'
    
    df.loc[CON_iv_dn_top_p_up_top_cp, 'play']='LP'
    df.loc[CON_iv_dn_top_p_up_lo, 'play']='LP'
    df.loc[CON_iv_dn_top_p_dn_top, 'play']='LP'
    df.loc[CON_iv_dn_top_p_dn_lo, 'play']='LC'
    
    df.loc[CON_iv_up_lo_dn_p_up_top, 'play']='BW,CAL'
    df.loc[CON_iv_up_lo_dn_p_dn_top, 'play']='LPV'
    
#intel_stat
    c1=df[CON_iv_up_top_p_up_top].shape[0]
    c2=df[CON_iv_up_top_p_up_lo].shape[0]   
    c3=df[CON_iv_up_top_p_dn_top].shape[0]
    c4=df[CON_iv_up_top_p_dn_lo].shape[0]   
    
    c5=df[CON_iv_dn_top_p_up_top_cp].shape[0]
    c6=df[CON_iv_dn_top_p_up_lo].shape[0]   
    c7=df[CON_iv_dn_top_p_dn_top].shape[0]
    c8=df[CON_iv_dn_top_p_dn_lo].shape[0]  

    c9=df[CON_iv_up_lo_dn_p_up_top].shape[0]
    c10=df[CON_iv_up_lo_dn_p_dn_top].shape[0]   

    print(" --- mc_intel stat: --  ")
    stat_mc={'date': q_date, 'c1':c1, 'c2':c2,'c3':c3, 'c4':c4, 'c5':c5, \
             'c6':c6, 'c7':c7, 'c8':c8, 'c9':c9, 'c10':c10} 
    df_stat=pd.DataFrame(stat_mc, index=np.arange(len(stat_mc)))
#    to_sql_append(df_stat, 'tbl_stat_mc')
    tbl_nodupe_append(q_date, df_stat, "tbl_stat_mc")
    print("stat_mc: \n", stat_mc)
#REDUCE candy number    
    du=df[pd.notnull(df.play)]
    du['earn_dt']=pd.to_datetime(du['earn_dt'])
#    for index, row in du.iterrows():
#        try:
#            du.loc[index,'earn_dt']=get_earning_ec(row.ticker)
#        except:
#            pass
#        try:
#            du.loc[index,'ex_div']=get_DIV(row.ticker)[0]
#        except:
#            pass
#        try:
#            du.loc[index, 'si']=get_si_sqz(row.ticker)[0]
#        except:
#            du.loc[index, 'si']=0
#        try:
#            du.loc[index, 'rsi']=int(float(get_rsi(row.ticker)))
#        except:
#            du.loc[index, 'rsi']=0
#            pass    
    

#    unop_show= ['ticker', 'play', 'v_pct', 'pct_c', 'iv_chg', 'iv', 'p_chg', 'p'\
#                ,'si','earn_dt', 'ex_div','event_dt', 'v_opt_avg']
    unop_show= ['ticker', 'play', 'v_pct', 'pct_c', 'iv_chg', 'iv', 'p_chg', 'p'\
                ,'earn_dt', 'event_dt', 'v_opt_avg']
     
    du=pd.concat([du[unop_show],du['event']], axis=1)
    candy= du.ticker.tolist()
#get Stat_view
    ds=stat_VIEW(q_date, candy)
    dus=pd.merge(du, ds, on='ticker', how='outer')
#Convert NaT to NaN
    dus['earn_dt']=pd.to_datetime(dus.earn_dt).dt.date
    dus.fillna('',inplace=True)
    stat_show=[ 'beta','srtn_22_pct', 'rtn_22_pct',  \
               'fm_50', 'fm_200','fm_hi','fm_lo']  #fm_mean50 is for ticker
    show= unop_show + stat_show       
    pd.set_option('display.expand_frame_repr', False)
  #save to tbl_mc_candy
    dusw=dus[show]
    dusw.sort_values(by=['v_pct','pct_c'], ascending=[False,False], inplace=True)
    print(" ----- mc_candy list ---- ")
    print(dusw)
    dusw['date']=q_date
    dusw['date']=pd.to_datetime(dusw['date'])
#    to_sql_append(dusw, 'tbl_mc_candy')    
    tbl_nodupe_append(q_date, dusw, 'tbl_mc_candy') 
     
#find repeat ticker
    repeat_days=5
    date_repeat=q_date - timedelta(repeat_days)
    qry="SELECT * FROM tbl_mc_raw where date>'%s'"%date_repeat
    dh=read_sql(qry, q_date)

    df_repeat=dh[dh.ticker.isin(dusw.ticker)]
    show_repeat=['ticker','pct_c','iv_chg','p_chg','p','event_dt','date' ]
#    df_repeat.drop(['Name','event','earn_dt'], axis=1, inplace=True)
    df_repeat=df_repeat[show_repeat]
    df_repeat.sort_values(['ticker','date'], ascending=[True,False], inplace=True)
    print(" ---  today mc_candy Occured in last %s days -----"%repeat_days)
    print(df_repeat)
#    plot_base(q_date, dus[show].ticker.unique().tolist(), dus[show])
    pd.set_option('display.expand_frame_repr', True)
    return df  #unfiltered        
    
def unop_bc(q_date): #q_date CRITIAL,must match unop_bc_date
 #Source:unop_bc file, tbl_pv_all, Dest: tbl_bc_raw
# https://www.barchart.com/options/unusual-activity
    print("  ---- unop_bc_dev starts -----  ")
    path='c:\\pycode\\eod'
    path=r"%s"%path
    files=os.listdir(path)
    files_unop=[f for f in files if f[:7]=='unusual']
    
    if len(files_unop)!=1:
        print("bc eod file not available or more than one file")
        return
    unop_date=files_unop[0][-14:-4]
    unop_date=parser.parse(unop_date).date()
    if unop_date!=q_date:
        print("unop_date not euqal to q_date")
        return
    df=pd.read_csv(r'c:\pycode\eod\%s'%files_unop[0])
    #last row is text useless
    df.drop(df.index[-1], inplace=True) 
    df.columns=df.columns.str.lower()
    df.rename(columns={'symbol':'ticker', 'exp date':'oexp_dt', 'open int':'oi'\
            ,'volume':'vol', 'vol/oi':'v_oi', 'time':'date'}, inplace=True)
#    date_fmt="{:%m/%d/%y}".format(q_date)) -not working as %s converted dt to str already!!
#    dbc=dbc[dbc.date== q_date.strftime("%m/%d/%y")] #get q_date data only
    df['ba_pct']= (df['last']-df['bid'])/(df['ask']-df['bid'])
    
    bid_ask_mark=0.75  #mark b or s, label for agreessivness, i.e. sweeping
    df.loc[df['ba_pct']>bid_ask_mark, 'bs']='b'
    df.loc[df['ba_pct']<(1-bid_ask_mark), 'bs']='s'
    
#add stk_vol, p_chg, v_stk_chg to dbc as raw input once only
    df['date']=pd.to_datetime(df['date'])
    max_pv_date=read_sql("SELECT max(date) FROM tbl_pv_all", q_date)
    max_pv_date=max_pv_date.iloc[0].values[0]
    max_pv_date=parser.parse(max_pv_date).date()
    
    o_date=q_date-datetime.timedelta(5)
    df_pv=read_sql("SELECT * FROM tbl_pv_all wHERE date>='%s'" %o_date, q_date)
    df_pv['date']=pd.to_datetime(df_pv['date']).dt.date
    df_pv=df_pv[df_pv.ticker.isin(df.ticker)][['ticker','date','close', 'volume']]
    list_last_2_dates=df_pv.date.unique()[-2:]
    list_last_date=[list_last_2_dates[-1]]
#date check
    if max_pv_date != q_date:
        print("unop_bc: max_pv_date not equal to q_date")
        return
    if list_last_date[0] !=max_pv_date:
        print("unop_bc: pv_all date inverse")
        return        
    
    df_pvp=df_pv[df_pv.date.isin(list_last_2_dates)]
    df_pvp=df_pvp.set_index(['date','ticker'])
    df_pvp.sort_index(axis=0, ascending=True, inplace=True)  
    df_pvp=df_pvp.groupby(level='ticker').pct_change().round(2)    
    df_pvp.dropna(inplace=True)
    df_pvp.columns=['p_chg','v_stk_chg']   
    df_pvp=df_pvp.reset_index()
    #volume
    df_pvv=df_pv[df_pv.date.isin(list_last_date)][['ticker','date','volume']]
    df_pvv.rename(columns={'volume':'stk_vol'}, inplace=True)   
    #merge p_chg, v_stk_chg, volume
    df_pvv['date']=pd.to_datetime(df_pvv['date'])
    df_pvp['date']=pd.to_datetime(df_pvp['date'])
    df_pvpv=df_pvv.merge(df_pvp,how='inner', on=['ticker','date'])
    df=df.merge(df_pvpv[['ticker','date','p_chg', 'v_stk_chg','stk_vol']], \
                     on=['ticker','date'],how='left')    
  
    tbl_nodupe_append(q_date, df, 'tbl_bc_raw')
    return df #enriched with p_chg, v_stk_chg, stk_vol but unfiltered orig.for unop_combo()

def bc_intel(q_date):
    print(" --bc_intel starts --")
    qry="SELECT * FROM tbl_bc_raw"
    df=read_sql(qry, q_date)
    df['date']=pd.to_datetime(df['date'])
    df=df[df.date==q_date]
    df.rename(columns={'price':'p'}, inplace=True)
    df['v_oi']=df['v_oi'].astype(float) 
    df['vol']=df['vol'].astype(float)
    df['dte']=df['dte'].astype(float)     
    df['ba_pct']=df['ba_pct'].astype(float)
    df['last']=df['last'].astype(float)  
    df['p']=df['p'].astype(float)  
#FILTERED BEGIN
    vol_oi_ratio=8
    vol_min=1500
    premium_min=0.2
    premium_max=5
    dte_min=2
    price_max=150
#    bid_ask_mark=0.8  #mark b or s, label for agreessivness, i.e. sweeping
#    
#    df.loc[df['ba_pct']>bid_ask_mark, 'bs']='b'
#    df.loc[df['ba_pct']<(1-bid_ask_mark), 'bs']='s'
    
    con_voloi=df['v_oi']>vol_oi_ratio
    con_vol=df['vol']>vol_min
    con_dte= df.dte>dte_min
    con_prem=(df['last']>premium_min) & (df['last']<premium_max)
    con_bs= pd.notnull(df['bs'])
    con_price=df['p']<price_max
               
    df=df[con_voloi & con_vol & con_dte & con_prem & con_bs & con_price]  #dbc filtered!!
    df.sort_values(['v_oi', 'vol', 'dte', 'iv', 'p'], \
        ascending=[False,False, False, True, True], inplace=True)
    df.drop_duplicates(['ticker','date'], keep='first', inplace=True)
    
    
    show_bc=['ticker','dte','type', 'bs', 'p_chg', 'v_stk_chg',\
     'strike', 'oexp_dt', 'last', 'iv', 'p', 'vol', 'oi', \
     'v_oi', 'date']
    
    pd.set_option('display.expand_frame_repr', False)
    print(" --- unop_bc unique ticker by  b/s, v_oi, dte, iv, price --- ")
    print (df[show_bc])
    pd.set_option('display.expand_frame_repr', True)
    try:
        df.drop('index', axis=1, inplace=True)
    except:
        pass

    tbl_nodupe_append(q_date, df, 'tbl_bc_candy')
    return df
    
# combine to tbl_bc_candy, no dupe
def bmc_intel(q_date): #dmc_raw and dbc_(unfilterd)
    dmc=read_sql("SELECT * FROM tbl_mc_raw", q_date)
    dbc=read_sql("SELECT * FROM tbl_bc_raw", q_date)
    dmc['date']=pd.to_datetime(dmc['date'])
    dbc['date']=pd.to_datetime(dbc['date'])
    dmc=dmc[dmc.date==q_date]
    dbc=dbc[dbc.date==q_date]
#    dmc.drop(['date','iv','p_chg'], axis=1, inplace=True)
    dbc.drop(['index', 'date','iv','p_chg'], axis=1, inplace=True)
    df=dmc.merge(dbc, on='ticker',how='left')  # mc_oriented
#FILTER REDUCE - exist in tbl_bc
    df=df[ pd.notnull(df.strike) ] 
#    show_bc=['bs', 'type', 'strike', 'oexp_dt', 'v_oi']
    show_bc=['ticker','dte','type', 'bs', 'p_chg', 'v_stk_chg',\
             'strike','oexp_dt', 'last', 'iv', 'vol', 'oi', 'v_oi', 'date']
    show_mc=['p', 'v_pct', 'pct_c', 'iv_chg','earn_dt','event_dt']
    
    show_bmc=show_bc + show_mc
    df=df[show_bmc]

    df['p']=df['p'].astype(float)
#    df['v_opt_avg']=df['v_opt_avg'].astype(float)
#    df['v_pct']=df['v_pct'].astype(float)
#    df['v_opt']=df['v_opt'].astype(float) 
            
# mc_intel  criteria (relax mc, keep bc)
#    volume_opt_ratio=2
#    volume_min=1000
#    p_chg_pct_top=3
#    iv_chg_pct_top=5
#    call_pct=75
#    put_pct=75
#    liq=1000  
    
#    con_p=(df.p<200) & (df.p>2)
#    con_liq=(df.v_opt_avg >liq)
#    con_vol_ratio=df.v_pct>volume_opt_ratio
#    con_vol_min= df.v_opt>volume_min
    
# mc_intel FILTER for less tickers
#    df=df[ con_p ]  
    
#bc_intel criteria
    vol_oi_ratio= 4#8
    vol_min=1500
    premium_min=0.2
    premium_max=5
    dte_min=2
    price_max=150

    df['v_oi']=df['v_oi'].astype(float)
    df['vol']=df['vol'].astype(float)
    df['dte']=df['dte'].astype(float)
    df['last']=df['last'].astype(float) 
    
    con_voloi=df['v_oi']>vol_oi_ratio
    con_vol=df['vol']>vol_min
    con_dte= df.dte>dte_min
    con_prem=(df['last']>premium_min) & (df['last']<premium_max)
    con_bs= pd.notnull(df['bs'])
    con_price=df['p']<price_max
               
    df=df[con_voloi & con_vol & con_dte & con_prem & con_bs & con_price]    

    df.sort_values(['v_pct','pct_c','ticker'], ascending=[False,False, True], inplace=True)

    df.fillna('',inplace=True)
    show_less=['v_oi','date','v_pct']
    show=[x for x in show_bmc if x not in show_less]
    pd.set_option('display.expand_frame_repr', False)
    print (" --- bmc_candy: mc_raw + bc_raw (relaxed)  ------ ")
    print (df[show])
    pd.set_option('display.expand_frame_repr', True)
    tbl_nodupe_append(q_date, df, 'tbl_bmc_candy')
    return df



def candy_track(q_date):
#tds[[']]
#1. tbl_candys (6 days) price confimration
#2. tbl_spec (6 days) price movement
#2 prem_b_s    
    print("-- candy_track started --")    
    days_track=6
    p_mv_min=0.015  #confirmation criteria
    v_mv_min=2
    pct_c_min=70
    p_date=q_date-datetime.timedelta(days_track)
    df_pv=read_sql("SELECT * FROM tbl_pv_all where date='%s'"%q_date, q_date)
#tbl_spec: p_mv, v_mv
    ds=read_sql("SELECT * FROM tbl_spec_candy", q_date)
    ds['date']=pd.to_datetime(ds['date'])
    ds=ds[ds.date>p_date]
    #exclude live trade
    dt=read_sql("SELECT * FROM tbl_c", q_date)
    list_track=list(set(ds.ticker) - set(dt.ticker))
    ds=ds[ds.ticker.isin(list_track)]
    ds=ds.merge(df_pv[['ticker','close','volume']],on=['ticker'], how='left')
    ds['p_mv']=ds['close'].astype(float)/ds['p'].astype(float)-1
    ds['v_mv']=ds['volume'].astype(float)/ds['stk_vol'].astype(float)-1
    con_mv_p=np.abs(ds['p_mv'])>=p_mv_min
    con_mv_v=ds['v_mv']>v_mv_min
    ds=ds[con_mv_p | con_mv_v]
    show_spec=['ticker', 'lsnv', 'pct_c', 'v_pct','p', 'earn_dt', 'iv',\
        'type', 'strike','vol', 'oexp_dt', 'v_oi', 'bs', 'note']
    ds=ds[show_spec] 
    print("\n candy_track: spec_candy p_m, v_mv ( past %s days) :\n"%days_track, ds)
#    plot_base(q_date, ds.ticker.unique(),ds)
    
    
#tbl_candies: p_mv,  
    df=read_sql("SELECT * FROM tbl_candies", q_date)
    df['date']=pd.to_datetime(df['date'])
    df=df[df.date > p_date]
    
    list_candy=list(set(df.ticker) - set(ds.ticker))    
    df=df[df.ticker.isin(list_candy)]
    df=df.merge(df_pv[['ticker','close']],on=['ticker'], how='left')
    df['p_mv']=df['close']/df['p'].astype(float)-1
    df['oexp_dt']=pd.to_datetime(df['oexp_dt'])
#    con_not_expire=( df['oexp_dt']>q_date )
#    df=df[con_not_expire]
    con_bull_mc=df['pct_c']> pct_c_min
    con_bull_bc1=(df['type'].str.lower()=='put')&(df['bs']=='s')    
    con_bull_bc2=(df['type'].str.lower()=='call')&(df['bs']=='b')
    CON_bull=con_bull_mc | con_bull_bc1 |con_bull_bc2
    con_bear_mc=df['pct_c']< (100- pct_c_min)  
    con_bear_bc1=(df['type'].str.lower()=='put')&(df['bs']=='b')
    con_bear_bc2=(df['type'].str.lower()=='call')&(df['bs']=='s')
    CON_bear= con_bear_mc | con_bear_bc1  |con_bear_bc2
#Continue track price movement confirmation        
    con_up=df['p_mv']>p_mv_min
    con_dn=df['p_mv']<-p_mv_min
    df.loc[CON_bull & con_up,'conf_p']=1
    df.loc[CON_bull & con_dn,'conf_p']=-1          
    df.loc[CON_bear & con_dn,'conf_p']=1     
    df.loc[CON_bear & con_up,'conf_p']= -1

    df.sort_values(['conf_p','date','bs'], \
        ascending=[False,False,True], inplace=True)          
#    con_show=pd.notnull(df['conf_p']) 
    con_show= df['conf_p']==1
    show_move=['ticker', 'conf_p','p_mv','pct_c','oexp_dt','strike'\
          ,'type','bs']
    df=df[con_show][show_move]
    df.fillna('',inplace=True)

    pd.set_option('display.expand_frame_repr', False)
    print("\n candy_intel: price confirmed for past %s days candy:\n"%days_track, df)
    pd.set_option('display.expand_frame_repr',True)
#    plot_base(q_date, df.ticker.unique(),df)
    return df
    
def candy_intel_orig(q_date):
# 1. consolidate mc_candy, bc_candy, bmc_candy into one table tbl_candies in past x days
#2. volume confimration candyvolume change in next x days
#Daily replacce "tbl_bc_pv"
# prem_b_s    
#    df_mc=read_sql("SELECT * FROM tbl_mc_raw WHERE date='%s'"%q_date)
#    df_bc=read_sql("SELECT * FROM tbl_bc_raw WHERE date='%s'"%q_date)
##stk_vol confirm same day or max 3 day
    print("-- candy_intel started --")    
    days_track=6
    p_date=q_date-datetime.timedelta(days_track)
    
    dmct=read_sql("SELECT * FROM tbl_mc_candy", q_date)
    dmct['date']=pd.to_datetime(dmct['date'])
    dmct=dmct[dmct.date > p_date]
    
    dbct=read_sql("SELECT * FROM tbl_bc_candy", q_date)
    dbct['date']=pd.to_datetime(dbct['date'])
    dbct=dbct[dbct.date >p_date]
    
    dbmct=read_sql("SELECT * FROM tbl_bmc_candy", q_date)
    dbmct['date']=pd.to_datetime(dbmct['date'])
    dbmct=dbmct[dbmct.date > p_date]

    df=pd.concat([dmct, dbct, dbmct])
    df=df[dbmct.columns]
    df.sort_values(['ticker','strike'], ascending=True, inplace=True)
    df.drop_duplicates(['ticker','date'], keep='first', inplace=True)
    df['oexp_dt']=pd.to_datetime(df['oexp_dt'])
#Update latest price 
    df_pv=read_sql("SELECT * FROM tbl_pv_all where date='%s'"%q_date, q_date)
    df=df.merge(df_pv[['ticker','close']],on=['ticker'], how='left')
    
    df['p_mv']=df['close']/df['p'].astype(float)-1

    p_mv_min=0.02
    v_pct_min=4
    pct_c_min=70
        
    CON_v_pct=df.v_pct>v_pct_min
    con_days_track=(q_date - pd.to_datetime(df['date']).dt.date)<datetime.timedelta(days_track)
    con_not_expire=( df['oexp_dt']>q_date )
    CON_date=con_days_track & con_not_expire
    
    con_bull_1=df['pct_c']> pct_c_min
    con_bull_2=(df['type'].str.lower()=='put')&(df['bs']=='s')    
    con_bull_3=(df['type'].str.lower()=='call')&(df['bs']=='b')
    CON_bull=con_bull_1 | con_bull_2 |con_bull_3

    con_bear_1=df['pct_c']< (100- pct_c_min)  
    con_bear_2=(df['type'].str.lower()=='put')&(df['bs']=='b')
    con_bear_3=(df['type'].str.lower()=='call')&(df['bs']=='s')
    CON_bear= con_bear_1 | con_bear_2  |con_bear_3


#    df_candy=df_candy[CON_v_pct | (con_bull_1 | con_bear_1)]

#Continue track price movement confirmation        
    con_up=df['p_mv']>p_mv_min
    con_dn=df['p_mv']<-p_mv_min
      
    df.loc[CON_bull & con_up,'conf_p']=1
    df.loc[CON_bull & con_dn,'conf_p']=-1          
    df.loc[CON_bear & con_dn,'conf_p']=1     
    df.loc[CON_bear & con_up,'conf_p']= -1
          
#Consolidate into one tbl_candies for q_date
    df_candy=df[df.date==q_date]
    

    con_show=pd.notnull(df['conf_p']) 
    show_move=['ticker', 'conf_p','p_mv','pct_c','iv','iv_chg','p_chg','close','oexp_dt','strike',\
          'dte','type','bs','earn_dt','event_dt','date']
    show_candy=[x for x in show_move if x not in ['conf_p','close','p_mv']]
    df_candy=df_candy[show_candy]
    ds=df[con_show][show_move]
    ds.fillna('',inplace=True)
    ds.sort_values(['conf_p','date','bs'], \
        ascending=[False,False,True], inplace=True)
#    df.sort_values(['date','#    
    pd.set_option('display.expand_frame_repr', False)
    print("candy_intel: all candies \n", df_candy)
#    to_sql_append(df_candy, 'tbl_candies')
    tbl_nodupe_append(q_date, df_candy, 'tbl_candies')
    print("\n candy_intel: price confirmed for past 6 days candy:\n", ds)
    pd.set_option('display.expand_frame_repr',True)
    plot_base(q_date, ds.ticker.unique(),ds)
    return ds

def get_unop_mc():  # take max 5 pages from unop  url
#ref: http://yizeng.me/2014/04/08/get-text-from-hidden-elements-using-selenium-webdriver/
    from lxml import html
    from selenium.webdriver.chrome.options import Options
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.keys import Keys
    from selenium.common.exceptions import TimeoutException
    from bs4 import BeautifulSoup

    chrome_options=Options()
    chrome_options.add_argument("--disable-popup")
    chrome_options.add_extension(r"c:\pycode\Github\extension_1_0_7_overlay_remove.crx")
    chrome_options.add_extension(r"c:\pycode\Github\extension_1_13_8.crx")  #fairad
    #chrome_options.add_extension(r"G:\Trading\Trade_python\pycode\Github\extension_0_3_4.crx")
    chrome_options.add_argument('--always-authorize-plugins=true')
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--start-maximized")  #full screen
    x=r"C:\Users\jon\AppData\Local\Google\Chrome\User Data\Default"
    chrome_options.add_argument("user-data-dir=%s"%x)
    url="https://marketchameleon.com/Reports/UnusualOptionVolumeReport"
    gecko="c:\\pycode\Github\chromedriver.exe"
    driver=webdriver.Chrome(executable_path="c:\pycode\Github\chromedriver.exe", \
        chrome_options=chrome_options)
    driver.get(url)

    time.sleep(3)
    html=driver.page_source
    soup=BeautifulSoup(html, 'lxml')
 #close flyer
    try:
        #flyer=soup.find("div",{"class":"register-flyer-close"})
        flyer=driver.find_element_by_css_selector('#register_flyer > div.register-flyer-close')
        flyer.click()
        flyer2=driver.find_element_by_id('register_flyer')
        flyer2.click()
    except:
        pass  
#find report date
    try:
        mc_date=soup.findAll("div",{"class":"report_as_of_time"})[0].text[13:]
    except: 
        mc_date=datetime.date.today().strftime("%Y-%m-%d")
 
#find subtitle
    titles=[]
    subs=soup.find("tr",{"class":"sub_heading"})
    for th_tags in subs.find_all('th'):
        titles.append(th_tags.get_text())

#find table contents
    table=soup.find("table",{"id":"opt_unusual_volume"})
    pages=[]
    pages.append(titles)   # APPEND whole list of subtile as first element
    aray=[]
    lis=[]
    for row in table.find_all('tr'):
        for td_tags in row.find_all('td'):
            x= td_tags.get_text()
            lis.append(x)
        aray.append(lis)
        lis=[]
    pages.extend(aray)
    nxt=driver.find_element_by_css_selector("a.paginate_button.next")
#    nxt_dis=driver.find_element_by_css_selector("a.paginate_button.next.disabled")
    page_num=0
    nxt_dis=False
    while (nxt and page_num<10 and (not nxt_dis)):
#        if page_num<10:
            try:
                nxt.send_keys(Keys.END)
            except:
                pass
    #        driver.execute_script("window.scrollTo(0,  document.body.scrollHeight);")
            time.sleep(2)
            try:
                nxt.click()
            except:
                driver.close()
                return pages, mc_date

            time.sleep(3)
            html=driver.page_source
            soup=BeautifulSoup(html, 'lxml')
            table=soup.find("table",{"id":"opt_unusual_volume"})
            aray=[]
            lis=[]
            for row in table.find_all('tr'):
                for td_tags in row.find_all('td'):
                    x= td_tags.get_text()
                    lis.append(x)
                aray.append(lis)
                lis=[]
            pages.extend(aray)
    #        next=driver.find_element_by_id("opt_unusual_volume_next")
            page_num+=1
            nxt=driver.find_element_by_css_selector("a.paginate_button.next")
            
            try:
                nxt_dis=driver.find_element_by_css_selector("a.paginate_button.next.disabled")
            except:
                pass
#        else:
#            driver.close()
#            return pages, mc_date

    driver.close()
    driver.quit()
    mc_date=parser.parse(mc_date).date()
    return pages, mc_date  #mc_date is datetime.date

def get_option_simple(ticker=''):
#ref: http://yizeng.me/2014/04/08/get-text-from-hidden-elements-using-selenium-webdriver/
    from lxml import html
    from selenium.webdriver.chrome.options import Options
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.keys import Keys
    from selenium.common.exceptions import TimeoutException

    chrome_options=Options()
    chrome_options.add_argument("--disable-popup")
    chrome_options.add_extension(r"c:\pycode\Github\extension_1_0_7_overlay_remove.crx")
    chrome_options.add_extension(r"c:\pycode\Github\extension_1_13_8.crx")  #fairad
    #chrome_options.add_extension(r"G:\Trading\Trade_python\pycode\Github\extension_0_3_4.crx")
    chrome_options.add_argument('--always-authorize-plugins=true')
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--start-maximized")  #full screen
    x=r"C:\Users\jon\AppData\Local\Google\Chrome\User Data\Default"
    chrome_options.add_argument("user-data-dir=%s"%x)
    url_1="https://marketchameleon.com/Overview/"+ticker
    url_2=url_1+"/OptionSummary/"
    url_3=url_1+"/StockPosts/"
    url_4=url_1+"/SymbolReview/"
    gecko="c:\\pycode\Github\chromedriver.exe"
    driver=webdriver.Chrome(executable_path="c:\pycode\Github\chromedriver.exe", \
        chrome_options=chrome_options)
#LOG IN
    url_0="https://marketchameleon.com/Account/Login"
#    driver.get(url_0)
#    time.sleep(4)
#    try:
#        ol=driver.find_element_by_class_name("register-flyer-close")
#        ol=driver.find_element
#        ol.click()
#    except:
#        print("no find flyer close")
#        pass
#    username = driver.find_element_by_id("UserName")
#    password = driver.find_element_by_id("Password")
#    username.send_keys("freeofjon@gmail.com")
#    password.send_keys("kevin2008")
#    submit=driver.find_element_by_xpath('//input[@class="site-login-control btn"]')
#    time.sleep(2)
#    submit.send_keys('\n')

#go to url_1
    driver.get(url_1)
    time.sleep(1)
    try:
        ol=driver.find_element_by_class_name("register-flyer-close")
        ol=driver.find_element
        ol.click()
    except:
        pass
    iv_30_rank=driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[3]/div[6]/span[2]').get_attribute('textContent')
    iv_30s=driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[3]/div[5]/span[2]').get_attribute('textContent')
    hv_1y=driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[3]/div[4]/span[2]').get_attribute('textContent')
    hv_22=driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[3]/div[3]/span[2]').get_attribute('textContent')

    v_stk=driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[2]/div[2]/span[2]').get_attribute('textContent')
    v_stk_avg=driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[2]/div[3]/span[2]').get_attribute('textContent')
    v_opt=driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[2]/div[4]/span[2]').get_attribute('textContent')
    v_opt_avg=driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[2]/div[5]/span[2]').get_attribute('textContent')
    v_stk=v_stk.replace(",","")
    v_stk_avg=v_stk_avg.replace(",","")
    v_opt=v_opt.replace(",","")
    v_opt_avg=v_opt_avg.replace(",","")
    v_stk_pct='{:.1%}'.format(float(v_stk)/float( v_stk_avg))
    try:
        v_opt_pct='{:.1%}'.format(float(v_opt)/ float(v_opt_avg))
    except:  #v_op is zero
        pass
#    etf_o=driver.find_element_by_xpath('//*[@id="mkt_corr_tbl"]/tbody/tr[1]/td[1]/span/a').get_attribute('textContent')
#    corr_etf=driver.find_element_by_xpath('//*[@id="mkt_corr_tbl"]/tbody/tr[1]/td[4]').get_attribute('textContent')
#    p_chg_etf=driver.find_element_by_xpath('//*[@id="mkt_corr_tbl"]/tbody/tr[1]/td[7]/span').get_attribute('textContent')
#    p_chg=driver.find_element_by_xpath('//*[@id="sym_heading_top"]/div[1]/div[2]/div[2]/p[3]').get_attribute('textContent')
#    p=driver.find_element_by_xpath('//*[@id="overview_last_price"]').get_attribute('textContent')
#    earn_date= driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[4]/div[4]/span[2]').get_attribute('textContent')
    ex_div= driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[4]/div[2]/span[2]').get_attribute('textContent')
    yld= driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[4]/div[3]/span[2]').get_attribute('textContent')
#    pe= driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[4]/div[5]/span[2]').get_attribute('textContent')


    iv_30=float(iv_30s.split(" ")[0])
    iv_30_chg=float(iv_30s.split(" ")[1])
    iv_30_chg='{:.1%}'.format(iv_30_chg/ iv_30)
    iv_30_rank=iv_30_rank.split(" ")[0]
    iv_hv=float(iv_30)/float(hv_22)
    hv_rank=float(hv_22)/float(hv_1y)

#go to url_2: option summary for call, put volume
    driver.get(url_2)
    time.sleep(1)
    try:
        ol=driver.find_element_by_class_name("register-flyer-close")
        ol=driver.find_element
        ol.click()
    except:
        pass

    v_c= driver.find_element_by_xpath('//*[@id="option_summary_MainTotal"]/tbody/tr/td[6]').get_attribute('textContent')
    v_p= driver.find_element_by_xpath('//*[@id="option_summary_MainTotal"]/tbody/tr/td[7]').get_attribute('textContent')
    oi_c= driver.find_element_by_xpath('//*[@id="option_summary_MainTotal"]/tbody/tr/td[8]').get_attribute('textContent')
    oi_p= driver.find_element_by_xpath('//*[@id="option_summary_MainTotal"]/tbody/tr/td[9]').get_attribute('textContent')

    v_c=float(v_c.replace(",", ""))
    v_p=float(v_p.replace(",", ""))
    oi_c=float(oi_c.replace(",", ""))
    oi_p=float(oi_p.replace(",", ""))
    pcr_v=v_p/ v_c
    pcr_oi=oi_p/ oi_c #avoid division by zero

    names=['iv_30', 'iv_30_rank', 'iv_hv', 'hv_1y', 'hv_22', 'hv_rank',   \
        'v_stk', 'v_stk_avg', 'v_opt', 'v_opt_avg', 'v_stk_pct','v_opt_pct', \
        'ex_div', 'yld', 'pcr_v', 'pcr_oi']
    values=[iv_30, iv_30_rank, iv_hv,  hv_1y, hv_22, hv_rank,\
            v_stk, v_stk_avg, v_opt, v_opt_avg, v_stk_pct, v_opt_pct, \
       ex_div, yld, pcr_v, pcr_oi]

    do=pd.DataFrame(data=[], columns=names)
    do.loc[0,names]=values
    do['ticker']=ticker
    driver.quit()
#    driver.close
    return do    
def download():
    import re
    import requests
    from bs4 import BeautifulSoup

    link = 'https://www.barchart.com/options/unusual-activity'

    r = requests.get(link)
    soup = BeautifulSoup(r.text, "html.parser")

    for i in soup.find_all('a', {'class': "toolbar-button download"}):
    #    print(re.search('http://.*\.apk', i.get('href')).group(0))
         print(i.get('href'))
         return i

def download_bc():
    from lxml import html
    from selenium.webdriver.chrome.options import Options
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.keys import Keys
    from selenium.common.exceptions import TimeoutException

    chrome_options=Options()
#    chrome_options.add_argument("--disable-popup")
    chrome_options.add_extension(r"c:\pycode\Github\extension_1_0_7_overlay_remove.crx")
#    chrome_options.add_extension(r"c:\pycode\Github\extension_1_13_8.crx")  #fairad
    #chrome_options.add_extension(r"G:\Trading\Trade_python\pycode\Github\extension_0_3_4.crx")
    chrome_options.add_argument('--always-authorize-plugins=true')
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--start-maximized")  #full screen
    x=r"C:\Users\jon\AppData\Local\Google\Chrome\User Data\Default"
    chrome_options.add_argument("user-data-dir=%s"%x)
    url="https://www.barchart.com/options/unusual-activity"

    gecko="c:\\pycode\Github\chromedriver.exe"
    driver=webdriver.Chrome(executable_path="c:\pycode\Github\chromedriver.exe", \
        chrome_options=chrome_options)
#LOG IN
    url_0="https://www.barchart.com/login"
    driver.get(url_0)
    time.sleep(4)
    try:
        ol=driver.find_element_by_class_name("register-flyer-close")
        ol=driver.find_element
        ol.click()
    except:
#        print("no find flyer close")
        pass
#    username = driver.find_element_by_id("bc-login-form")
#    password = driver.find_element_by_id("login-form-password")

    try:
        username=driver.find_element_by_xpath('//*[@id="bc-main-content-wrapper"]/div/div[2]/div[2]/div/div/div/div[2]/form/div[1]/input')
        password=driver.find_element_by_xpath('//*[@id="login-page-form-password"]')
        username.send_keys("boxofjon@yahoo.com")
        password.send_keys("kevin2008")
        submit=driver.find_element_by_xpath('//*[@id="bc-main-content-wrapper"]/div/div[2]/div[2]/div/div/div/div[2]/form/div[4]/button')
        time.sleep(2)
        submit.send_keys('\n')
    except:
        pass  #if user login cache preserved

    driver.get(url)
    time.sleep(4)
    download=driver.find_element_by_xpath('//*[@id="main-content-column"]/div/div[3]/div[2]/div[2]/a[4]')
    download.click()
    time.sleep(10)
#    download.send_keys(Keys.ENTER)
#    download.send_keys('\n')
    print("i am waiting")
    driver.close()
    driver.quit()  

'''
1. sentiment on volume (not premium!)
   mc: vol_p, vol_c; 
   bc (new postition premium!): vol_p_b, vol_p_s, vol_c_b(), vol_c_s(bwrite)
   
2. vol_stk confirm (same day or 3 day + )
3. 
Decipline:
a. tgt_dt, tgt_p (catalyst_dt) -> give SPEC, if term=S, then out if no gain in x days
b. ibq/e/t (by term)
'''