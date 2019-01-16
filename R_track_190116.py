# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np 
import datetime as datetime
import sqlite3 as db
from timeit import default_timer as timer

from termcolor import colored, cprint
import warnings
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.2f}'.format
from P_commons import to_sql_append, to_sql_replace, read_sql

'''
1. entry to tbl_candy
2. read to csv, update to tbl_trd
3. daily: track_raw
4. daily: track_derived
5. daily: track_alert
6. daily: track_dash
'''
todate=datetime.date.today()
def main_TRACK(q_date):
    print("trade_track started")    
    dt=track_data(q_date)
    if not dt.empty:
        df=track_show(dt)
    print("trd_track is done")

#get candy_list details from  previous 7 days of tbl_mc_candy, tbl_spec_candy
#then update by Flask_trd_edit, then append to tbl_c
def candy_to_trd(q_date, candy_tickers=['SPEC']):
    candy_days=10
    candy_date=datetime.datetime.today()-datetime.timedelta(candy_days)  #whitin 7 days
    df=read_sql("SELECT * FROM tbl_mc_candy WHERE date>'s'"%candy_date,q_date)
    ds=read_sql("SELECT * FROM tbl_spec_candy WHERE date>'s'"%candy_date,q_date)
    df=pd.merge(df, ds, on=['ticker','date'], how='outer')
    df=pd.merge(df, )
    con_trd=df.ticker.isin(candy_tickers) & (pd.to_datetime(df['date'])>candy_date) 
#narrow down candy list     
    df=df[con_trd]  
    df.rename(columns={'p':'entry_p','ex_div':'div_dt',\
            'srtn_22_pct':'i_srtn_22_pct', 'rtn_22_pct':'i_rtn_22_pct',\
            'rsi':'i_rsi'}, inplace=True)
    df['i_mc']=df['v_pct'].astype(str)+'|'+df['pct_c'].astype(str)

    show_candy=['ticker','entry_p', 'earn_dt', 'div_dt', 'i_rsi', 'i_srtn_22_pct',\
       'i_rtn_22_pct', 'beta', 'fm_50', 'fm_200', 'fm_hi','fm_lo', 'event', 'date', 'si',\
       'play', 'i_mc']
    candy_extra=['act', 'term', 'play', 'comm', 'strike','exp_dt','last_p',\
        'iiv', 'beup', 'bedn', 'delta','vega','note',\
        'ip1','ic1','ip2','ic2','op1','oc1','op2','oc2',\
        'tgt_p','tgt_dt', 'entry_dt','exit_dt', 'mp']
    df_extra=pd.DataFrame(columns=candy_extra, index=df.index)
    dc=pd.concat([df[show_candy], df_extra])
    dc.replace(np.NaN, '', inplace=True)
    dc=dc[dc.ticker!='']
#pre-fill rsi?
    dc['exit_dt']='N'
    dc['act']='IBQ'
    to_sql_replace(dc,"tbl_trade_candy")
    dc.to_csv('c:\\pycode\playlist\candy_%s.csv'%(todate)) 
    return dc

def trd_to_table():
#    df_trade=pd.read_excel(open(r'c:\pycode\playlist\candy.xlsx','rb'))
    df_trade=pd.read_csv(open(r'c:\pycode\playlist\candy.csv'))
    if  'Unnamed: 0' in (df_trade.columns):
        df_trade.drop('Unnamed: 0', axis=1, inplace=True)
    to_sql_append(df_trade, 'tbl_c')  #log trades and track performance
    print ("trade is saved in tbl_c")

def track_raw(q_date): 
    '''
1  update ['exit_dt, lp, note, iiv, oiv] on tbl_trd (Flask)
2. Read from tbl_trd for live ticker (for non_live save to tbl_hist)
3  Read from tbl_track for Non-live ticker, save to tbl_track_hist (Note: one ticker only in tbl_trade)
4. Auto_update from stat_VIEW, tbl_mc_raw on EXACT q_date daily
5  Flask update to tbl_c 

!! to do:

    3. track_raw():
        tbl_trd/histL non-live: append to trd_hist; write back to tbl_trd
        live: read tbl_trd (live only)
        auto_update: from stat_VIEW, tbl_mc_raw, tbl_bc_raw, 
        display@Flask??
        '''
    from R_stat import stat_VIEW, stat_run_base
    from T_intel import get_earnings_ec
    dt_all=read_sql("SELECT * FROM tbl_c", q_date)
#update cap data
    df_cap=read_sql("SELECT * FROM tbl_cap", q_date)
    dt_all=dt_all.merge(df_cap[['act','cap']], on='act',how='left')
    if set(['level_0']).issubset(dt_all.columns):    
        dt_all=dt_all.drop(['level_0'],axis=1)
    dt_hist=dt_all[dt_all.exit_dt != 'N']
    to_sql_append(dt_hist, 'tbl_c_hist') 
# REPLACE live trade to tbl_trd   
    dt=dt_all[dt_all.exit_dt=='N']
    dt=dt.fillna(0)
    con_nullticker=pd.isnull(dt.ticker)
    i_nullticker=dt[con_nullticker].index
    dt=dt.drop(i_nullticker)
    if dt.shape[0]==0:  #prevent erase tbl_c
        print("dt empty")
        return
    to_sql_replace(dt, "tbl_c")  
#UPDATE to current live, non-live trade
# UPDATE RAW from stat_view, len(dt)=len(ds) !!!       
    ds=stat_VIEW(q_date, dt_all['ticker'].tolist())
    ds.sort_values(['ticker'], ascending=True)
    dt.sort_values(['ticker'], ascending=True)
    con=ds['ticker'].isin(dt['ticker'])  #obtain index in ds
    dt['date']=q_date
#    loc_t=pd.isnull(dt.close)  #ticker not in sp or etf
#    loc_s=ds['ticker'].isin(dt.loc[loc_t,'ticker']) 
#raw fileds to update daily
    dt.drop(['fm_50', 'fm_200', 'fm_hi','fm_lo','last_p'], axis=1, inplace=True)  #avoid dupe fields with ds
    show_raw=['ticker','close','fm_hi','fm_lo','fm_50', 'fm_200',\
    'rsi','srtn_22_pct', 'rtn_22_pct']
#         'mc','bc']
    dt=dt.merge(ds[show_raw], on='ticker', how='left')
    dt.rename(columns={'close':'last_p'}, inplace=True)
#    for index, row in dt.iterrows():
#        opt=get_option_simple(row.ticker)
#        dt.loc[index, 'iv_30']=opt['iv_30'].values[0]
#        dt.loc[index,'iv_30_rank']=float(opt['iv_30_rank'].values[0][:-1])
#        dt.loc[index,'v_opt_pct']=opt['v_opt_pct'].values[0]    
        
#    dt['iv_rank_chg']=dt['iv_30_rank']-dt['iv_30_rank_prev']
#    dt['std']=dt['close_qdate_prev']* dt['iv_30']*0.01*np.sqrt(30/365)
#    dt['spike']=(dt['close_qdate'] - dt['close_qdate_prev'])/dt['std']
#    dt.replace(np.NaN,0, inplace=True)
    dt['rsi_chg']=dt['rsi'].astype(float) - dt['i_rsi'].astype(float)
    dt['ts_chg'] = dt['rtn_22_pct'].astype(float) - dt['i_rtn_22_pct'].astype(float)
    dt['sm_chg'] = dt['srtn_22_pct'].astype(float) - dt['i_srtn_22_pct'].astype(float) 

# UPDATE from external, mc, bc
    df_mc=read_sql("SELECT * FROM tbl_mc_raw WHERE date='%s'"%q_date, q_date)
    loc_mc=dt['ticker'].isin(df_mc['ticker'])
    if loc_mc.any():
        dt.loc[loc_mc,'mc']=df_mc['ticker'] +'|' + df_mc['v_pct'].astype(str)\
            +'|'+df_mc['pct_c'].astype(str)
    else:
        dt['mc']=''
    dt['bc']=''
    
# UPDATE derived: ic, ip, oc, op, lc, lp, mp, risk_orig, risk_live, pl_r, pl_ur, pl_pct, 
    dt.replace("",0, inplace=True)
    dt['ic']= dt['ic1']+dt['ic2']
    dt['ip']=(dt['ic1']*dt['ip1']+dt['ic2']*dt['ip2'])/dt['ic']    
    dt['oc']= dt['oc1']+dt['oc2']
    dt['op']=(dt['oc1']*dt['op1']+dt['oc2']*dt['op2'])/dt['oc']  
    dt['lc']=dt['ic']- dt['oc']
    
    con_null_op=pd.isnull(dt['op']) #if not sold then lp=0 default
    dt.loc[con_null_op, 'op']=0 #dt.loc[con_null_op, 'ip']
    dt['lp']= (dt['ic']*dt['ip']-dt['oc']*dt['op'])/dt['lc']
# if mp is NOT updated by tbl_trd@Flask
    dt[pd.isnull(dt.mp)]['mp']=dt['lp']  

    dt['risk_orig']=dt['ic']*dt['ip']*100+dt['comm']
    dt['risk_live']=dt['lc']*dt['mp']*100
    dt['pl_r']=(dt['op']-dt['ip'])*dt['oc']*100
    dt['pl_ur']=(dt['mp']-dt['lp'])*dt['lc']*100
    dt['pl']=dt['pl_r'] + dt['pl_ur']- dt['comm']
    dt['pl_pct']=(dt['pl']/dt['risk_orig'])*100
    
    dt['days_to_exp']=pd.to_datetime(dt['exp_dt']).subtract(pd.to_datetime(dt['date'])).dt.days   
    dt['days_all']= pd.to_datetime(dt['exp_dt']).subtract(pd.to_datetime(dt['entry_dt'])).dt.days   
    dt['days_pct']=(1-dt['days_to_exp']/dt['days_all']).round(2)
    try:
        dt['days_to_div']=pd.to_datetime(dt['div_dt']).subtract(pd.to_datetime(dt['date'])).dt.days
    except:
        dt['days_to_div']=1000
    dt['earn_dt']= dt['earn_dt'].values[0].split()[0]
    dt['days_to_earn']=pd.to_datetime(dt['earn_dt']).subtract(pd.to_datetime(dt['date'])).dt.days
    dt['fm_strike']=(dt['last_p']/dt['strike']-1).round(2)
    dt['srtn_chg']=dt['srtn_22_pct']<dt['i_srtn_22_pct']
    dt['rtn_chg']=dt['rtn_22_pct']>dt['i_rtn_22_pct']
    dt['rsi_chg']=dt['rsi']<dt['i_rsi']
    try:
        dt['tgt_dt']=pd.to_datetime(dt['tgt_dt'])
    except:
        dt['tgt_dt']=datetime.datetime(2019,1,1).date()
    dt['weigh']=dt['risk_live']/dt['cap']
#Track @ Act & asset level

    z1=dt.groupby('act')['risk_live'].sum()    
    z2=dt.groupby('act')['weigh'].sum()
    z3=dt.groupby('act')['pl'].sum()
    
    z=list(zip(z1, z2, z3))
    dt_overview=pd.DataFrame(z,columns=['act', 'risk_live','weigh', 'pl'], index=z1.index)
    print (" ---- Track Overview ---- ")
    print(dt_overview)
    
## UPDATE Alert - variables
    stop_loss_pct= -0.5
    p_runaway_pct= 0.1
    v_stk_vol_pct=2.5
    key_level_pct=0.01    
    exit_days_pct=0.7
    exit_days_to_exp=5
    exit_pl_pct=0.3  #net of comm?
    event_days=5
    weigh_per_ticker=0.2
    buff=0.15
    fm_strike_pct=0.03
    dt['buff']=0
    
    lsnv=['L','S','N','V']
    pc=['P','C','PC']
    spreads=['V','CAL','BF','IC','BW','SYN']
    
#add field "way", "pc" to tbl_c and flask?    
    con_l=dt['lsnv'].astype('str').str.upper()=='L'
    con_s=dt['lsnv'].astype('str').str.upper()=='S'
    con_n=dt['lsnv'].astype('str').str.upper()=='N'
    con_v=dt['lsnv'].astype('str').str.upper()=='V'
    
    con_fm_strike_up=(dt['last_p']/dt['strike']-1)>=fm_strike_pct
    con_fm_strike_dn=(dt['last_p']/dt['strike']-1)<=(0 - fm_strike_pct)
# alert_itm:     
    con_sc_up=con_s & (dt['pc'].astype('str').str.upper()=='C') & (pd.isnull(dt['play'])) & con_fm_strike_up
    con_sp_dn=con_s & (dt['pc'].astype('str').str.upper()=='P') & (pd.isnull(dt['play'])) & con_fm_strike_dn
    con_c_spread_up= (dt['pc'].astype('str').str.upper()=='C') & (dt.play.isin(spreads)) & con_fm_strike_up
    con_p_spread_dn= (dt['pc'].astype('str').str.upper()=='P') & (dt.play.isin(spreads)) & con_fm_strike_dn  
    CON_itm=con_sc_up | con_sp_dn | con_c_spread_up |con_p_spread_dn                    
    
    con_momt_l= con_l & (  (dt['srtn_22_pct']<dt['i_srtn_22_pct']) \
                    | (dt['rtn_22_pct']<dt['i_rtn_22_pct']) \
                    | (dt['rsi']<dt['i_rsi'])  ) 
    con_momt_s= con_s & ( (dt['srtn_22_pct']>dt['i_srtn_22_pct']) \
                   | (dt['rtn_22_pct']>dt['i_rtn_22_pct']) \
                   | (dt['rsi']<dt['i_rsi']) )
    CON_momt= con_momt_l | con_momt_s

    CON_key_level=(np.abs(dt['fm_50'])<= key_level_pct) \
          |(np.abs(dt['fm_200'])<= key_level_pct) \
          |(np.abs(dt['fm_hi'])<= key_level_pct) \
          |(np.abs(dt['fm_lo'])<= key_level_pct)   

    CON_be=(dt.last_p>dt.beup.astype(float)) | (dt.last_p< dt.bedn.astype(float))
    CON_stop=((dt['mp']/dt['lp']-1)<= stop_loss_pct)
    
 #tgt_dt is the KEY control over SPEC risks
    CON_out=dt['tgt_dt']>q_date 
    CON_prof=dt['mp']>dt['tgt_p']
        
    con_exit_no_chance=((dt['pl_pct']<0 )&(dt['days_pct']>=exit_days_pct))
    con_exit_no_time= (dt['days_to_exp']<=exit_days_to_exp)
    con_exit_pl= dt['pl_pct']>exit_pl_pct
    CON_exit= con_exit_no_chance |con_exit_no_time |con_exit_pl

    CON_event=(dt['days_to_div']<=event_days)|(dt['days_to_earn']<=event_days)
    CON_runaway= (np.abs(dt['last_p']/dt['entry_p']-1)>=p_runaway_pct)
#    con_v=stk vol chg |pd.notnull(dt['mc']) |pd.notnull(dt['mc'])
    CON_ov=pd.notnull(dt['mc']) |pd.notnull(dt['bc'])  #unop for live trade !!
#    con_k= any(list(filter(lambda x:np.abs(x)<=k_level_pct, dt[['fm_hi','fm_lo','fm_50','fm_200']]))
#    con_k=any(np.abs(x)<=k_level_pct for x in dt[['fm_hi','fm_lo','fm_50','fm_200']])
    CON_weigh=dt['weigh'] >=weigh_per_ticker
    
#UPDATE alert
#    dt['buff']=1- dt.groupby('act')['risk_live'].sum()/dt['cap']
#    CON_buff=dt['buff']<=buff
    dt['a_out']=CON_out
    dt['a_stop']=CON_stop
    dt['a_prof']=CON_prof
    dt['a_itm']=CON_itm
    dt['a_event']=CON_event
    
    dt['a_be']=CON_be
    dt['a_exit']=CON_exit
    dt['a_ov']=CON_ov
    
    dt['a_key']=CON_key_level
    dt['a_run']=CON_runaway
    dt['a_momt']=CON_momt
    dt['a_weigh']=CON_weigh
#    dt['a_buff']=CON_buff
    dt.replace(False, "", inplace=True)
    dt.replace(np.nan, "", inplace=True)
    try:
        dt.drop(['index','Unnamed: 0'], axis=1, inplace=True)
    except:
        pass
#APPEND to tbl_track_hist   (record of full trade for Analysis)
    dt_track_hist=dt[dt.ticker.isin(dt_hist.ticker)]
    to_sql_append(dt_track_hist, "tbl_track_hist")
    
#REPLACE tbl_track with latet data    
    dt_track_live=dt[dt.ticker.isin(dt.ticker)]
    to_sql_replace(dt_track_live, "tbl_track")
    
    df=dt_track_live
    
#    show=['act','ticker','last_p','ip', 'lp', 'mp', 'risk_live','pl_pct', 'days_to_exp']
    show=['act', 'ticker','days_to_exp']
    show_alerts=['a_out', 'a_stop', 'a_prof', 'a_itm','a_event', 'a_exit', 'a_be','a_key', 'a_momt','a_run',\
               'a_ov', 'a_weigh']
    show_alerts=show+show_alerts
    show_base=['ticker','last_p','risk_live','pl_pct', 'tgt_dt', 'lp', 'mp','tgt_p','days_pct']
    show_out=show_base
    show_stop=show_base + ['bedn','beup','days_to_exp']
    show_prof=show_base
    show_exit=show_base + ['days_to_exp']
    show_itm=show_base +['lsnv','pc','play','strike']
    show_runaway_key_momt=show_base+['strike','entry_p','fm_50','fm_200','rtn_chg','srtn_chg','rsi_chg']
    show_ov_event=show_base+['div_dt','event','mc','bc']
    
    pd.set_option('display.expand_frame_repr', False)
    print(" ---- ALERT MATRIX  ---       ")
    print(df[show_alerts])    
    print(" ---- OUT tgt_dt (no catalyt)  ---       ")
    print(df[CON_out][show_out])
    print(" ---- stop loss ---       ")
    print(df[CON_stop][show_stop])
    print(" ---- Exit prof ---       ")
    print(df[CON_prof][show_prof])
    print("      ---- exit ---           ")
    print(df[CON_exit][show_exit]) 
    print("      ---- In the money ---           ")
    print(df[CON_itm][show_itm]) 
    print("          ---- runawy_key_momt ---   ")   
    print(df[CON_runaway | CON_key_level | CON_momt][show_runaway_key_momt])
    print("                   --- ov_event  -----")
    print(df[CON_ov |CON_event][show_ov_event])
    pd.set_option('display.expand_frame_repr', True)
    return df
# INTEL_op
# iv rank only
#2. web_mc: one year iv/ vwop data -> plot or ai
#3. p/c premium sentiment
#4