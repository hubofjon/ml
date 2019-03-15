# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np 
import datetime as datetime
import sqlite3 as db
from timeit import default_timer as timer
from R_plot import plot_base
from dateutil import parser

from termcolor import colored, cprint
import warnings
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.2f}'.format
from P_commons import to_sql_append, to_sql_replace, read_sql, type_convert
from P_commons import *
from R_stat_dev import stat_VIEW, stat_run_base

from T_intel import get_earning_ec, get_DIV, get_si_sqz, get_rsi, get_RSI
'''
1. entry to tbl_candy
2. read to csv, update to tbl_trd
3. daily: track_raw
4. daily: track_derived
5. daily: track_alert
6. daily: track_dash
pd.to_datetime(dfx, format='%d-%m-%Y', errors='coerce')
'''
todate=datetime.date.today()
def main_TRACK(q_date):
    print("trade_track started")    
    dt=track_data(q_date)
    if not dt.empty:
        df=track_show(dt)
    print("trd_track is done")

def spec_candy(q_date, spec_list=[]):
    '''
    same day ad_hoc from past 10 days spec_candy, as base for t_candy and flask update
    '''
    lookback=10
    p_date=q_date-datetime.timedelta(lookback)
    if len(spec_list)==0:
        print("empty spec_list")
        return
#reduct to spec_list tickers
    df=read_sql("SELECT * FROM tbl_candies where date>='%s'"%p_date, q_date)
    df=df[df.ticker.isin(spec_list)]
#if same ticker multi-occurence, keep the latest date
    df.sort_values(['ticker','date'], ascending=[True, False], inplace=True)
    df.drop_duplicates('ticker', keep='first',inplace=True)
    
#update the stat_VIEW section (for ticker wro 'sec','beta' info)
    ds=stat_VIEW(q_date, spec_list, 'update') 
    col_ds=ds.columns.tolist()
    col_df=df.columns.tolist()
    col_nds=list(set(col_df)-set(col_ds))+['ticker']
    df_nds=df[col_nds]
    df=pd.merge(df_nds, ds, how='left', on='ticker')
#    df.loc[:,col_ds]=ds
    #extra field for flask_candy input & 
    df['note']=''
    df['lsn']=''
    df['bet']=''
    df['stk_vol']=''
    df.drop('index',axis=1, inplace=True)
    scm_t_spec=df.columns.tolist()
    df=df[scm_t_spec]
    to_sql_append(df,"tbl_spec_candy")
    print("tbl_spec_cany appended  %s"%q_date)
    return df

#Source: spec_candy past 10 days, Dest: replace tbl_trade_candy
def trade_candy(q_date, trade_list=['']):
    lookback=10
    trade_list=[x.upper() for x in trade_list]
    candy_date=datetime.datetime.today()-datetime.timedelta(lookback)  #whitin 7 days

    df=read_sql("SELECT * FROM tbl_spec_candy", q_date)
    
    df['date']=pd.to_datetime(df['date'])
    
    con_trd=(df.ticker.isin(trade_list)) & (df['date']> candy_date) 
#narrow down candy list     
    df=df[con_trd]  
    if df.empty:
        print ("trade_candy: not in tbl_spec_candy")
        return
    
#Update with stat_VIEW @entry date 
    t_list= df.ticker.tolist()
    ds=stat_VIEW(q_date, t_list) 
    col_ds=ds.columns.tolist()
    col_df=df.columns.tolist()
    col_nds=list(set(col_df)-set(col_ds))+['ticker']
    df_nds=df[col_nds]
    df=pd.merge(df_nds, ds, how='left', on='ticker')   
    
    df['date']=q_date
    
    df.drop('index',axis=1, inplace=True)
    scm_t_spec=df.columns.tolist()
    col_rn=['p','iv','hv_22','rtn_22_pct','srtn_22_pct', 'strike']
    col_rnto=['entry_p','i_iv','i_hv_22','i_rtn_22_pct','i_srtn_22_pct','strik']
    
    df.rename(columns=dict(zip(col_rn, col_rnto)), inplace=True)
 
    col_extra=['act', 'bedn', 'beup', 'comm', 'delta', 'entry_dt', \
        'exit_dt', 'exp_dt', 'ic1', 'ic2', 'ip1', 'ip2',\
        'close', 'mp', 'oc1', 'oc2', 'op1', 'op2',\
        'pc', 'play', 'strike', 'term', 'tgt_dt', 'tgt_p', 'vega']
    
    df_extra=pd.DataFrame(columns=col_extra, index=df.index)
    dc=pd.concat([df, df_extra], axis=1)
    scm_t_trdcdy=dc.columns.tolist()
    
#pre_fill values
    
    dc['exit_dt']='N'
    dc['act']='IBQ'    
    dc['entry_date']=q_date
    to_sql_replace(dc,"tbl_trade_candy")
    print(" tbl_trade_candy is replaced" )    

    show=['ticker','entry_p', 'earn_dt', 'div_dt', 'i_srtn_22_pct',\
       'i_rtn_22_pct',  'si', 'date','note']
#    dc.replace(np.NaN, '', inplace=True)
    print(dc[show])
    return dc

def trade_entry(q_date):
#    df_trade=pd.read_excel(open(r'c:\pycode\playlist\candy.xlsx','rb'))
#    df_trade=pd.read_csv(open(r'c:\pycode\playlist\candy.csv'))
    df=read_sql("SELECT * FROM tbl_trade_candy", q_date)
    dr=read_sql("SELECT * FROM tbl_rsik")
# data validation
    df['lsnv']=df['lsnv'].astype('str').str.upper()
    df['play']=df['play'].astype('str').str.upper()
    df['pc']=df['pc'].astype('str').str.upper()
    con_lsnv= df['lsnv'].isin(['L','S','N','V']).all()
    con_play= df['play'].isin(dr.play.tolist()).all()
    if  not ( con_lsnv & con_play):
        print("trade_entry: lsnv or play not in list")
        return    
    ask=input(" --- trade_entry?  yes/n:    ")  
    if ask=='yes':
        
#        if  'index' in (df.columns):
#            df.drop('index', axis=1, inplace=True)
        scm_t_c=read_sql("select * from tbl_c").columns.tolist()
        scm_t_c=[x for x in scm_t_c if x not in \
            ['index','level_0','cap','i_rsi','i_mc']]
        df=df[scm_t_c]
        to_sql_append(df, 'tbl_c')  #log trades and track performance
        print ("trade is appended in tbl_c")
        return df
# generate spec_candy from tbl_mc_raw, tbl_bc_raw on q_date

def spec_track(q_date):
#tbl_spec_candies: (in 6 days)track p_chg, stk_v_chg 
    days_trace=7
    p_date=q_date - datetime.timedelta(days_trace)
    df=read_sql("SELECT * FROM tbl_spec_candy", q_date)
    df['date']=pd.to_datetime(df['date'])
    df=df[df.date>p_date]
    #exclude live trade
    dt=read_sql("SELECT * FROM tbl_c", q_date)
    list_track=list(set(df.ticker) - set(dt.ticker))
    df=df[df.ticker.isin(list_track)]
    
    #get price/vol data
    
    show=['ticker', 'lsn', 'pct_c', 'v_pct','p', 'earn_dt', 'iv',\
        'type', 'strike','vol', 'oexp_dt', 'v_oi', 'bs', 'note']
    df=df[show]
    print("spec_track  %s  days plot "%days_trace)
    plot_base(q_date, df.ticker.unique(),df)
    return df

def t_convert(df, cols, type='float'):
    if type=='float':
        for x in cols:
            df[x]=df[x].astype(type)
            return df
    elif type=='datetime':
        dummy_dt='2000-1-1'
        for x in cols:
            df[x]=df[x].replace('',dummy_dt).astype(str)
            df[x]=df[x].replace('None', dummy_dt).astype(str)
            df[x]=df[x].apply(lambda i: parser.parse(i))
        return df 

def track_raw(q_date): 
    '''
    1. update flask with exit, mp
    2. update tbl_c, move exit to tbl_c_hist with pnl
    3. update tbl_c with stat_VIEW
    4. calc pnl, position, risk, 
    5. ask ? stop, exit, aging, event_dt, spike_qdate, hv_22 move, 
    6. algo- prob_R, prob_Rx vs. i_pr, iprx ()
    '''
#Source: tbl_c, stat_view, tbl_mc_raw, tbl_bc_raw, Append: tbl_c_hist, Replace:tbl_c
    '''
!! to do:
    3. track_raw():
        tbl_trd/histL non-live: append to trd_hist; write back to tbl_trd
        live: read tbl_trd (live only)
        auto_update: from stat_VIEW, tbl_mc_raw, tbl_bc_raw, 
        display@Flask??
        '''
    dt_all=read_sql("SELECT * FROM tbl_c", q_date)
 #Validate act, cap   
    dt_all['act']=dt_all['act'].str.upper()
    dt_all.drop('cap', axis=1, inplace=True)
#update cap data
    df_cap=read_sql("SELECT * FROM tbl_act", q_date)
    dt_all=dt_all.merge(df_cap[['act','cap']], on='act',how='left')
#    if set(['level_0']).issubset(dt_all.columns):    
#        dt_all=dt_all.drop(['level_0'],axis=1)
#    dt_hist=dt_all[dt_all.exit_dt != 'N']
#    to_sql_append(dt_hist, 'tbl_c_hist') 
    '''    
# REPLACE live trade to tbl_trd   
    dt=dt_all[dt_all.exit_dt=='N']
    dt=dt.fillna(0)
    con_nullticker=pd.isnull(dt.ticker)
    i_nullticker=dt[con_nullticker].index
    dt=dt.drop(i_nullticker)
    if dt.shape[0]==0:  #prevent erase tbl_c
        print("dt empty")
        return
    if not dt_hist.empty:  #only update tbl_c when needed
        to_sql_replace(dt, "tbl_c")  
    '''

# updte with latest stat_VIEW
    ds=stat_VIEW(q_date, dt_all['ticker'].tolist())
    col_share=list(set(dt_all.columns).intersection(set(ds.columns)))
    col_drop=[x for x in col_share if x not in ['ticker']]
    dt_all.drop(col_drop, axis=1, inplace=True)
    dt=pd.merge(dt_all, ds, on='ticker')
    
# in mc_raw or bc_raw?
    df_mc=read_sql("SELECT * FROM tbl_mc_raw ", q_date)
    df_mc['date']=pd.to_datetime(df_mc['date'])
    df_mc=df_mc[df_mc.date==q_date]
    loc_mc=dt['ticker'].isin(df_mc['ticker'])
    if loc_mc.any():
        dt.loc[loc_mc,'mc']=df_mc['ticker'] +'|' + df_mc['v_pct'].astype(str)\
            +'|'+df_mc['pct_c'].astype(str)
    else:
        dt['mc']=''
    dt['bc']=''
#Clean/convert data
#1. from flask 
    dt['date']=q_date
    dt['date']=pd.to_datetime(dt['date'])
    dummy_dt='2000-1-1'
    flask_val=['ic1','ip1','ic2','ip2','oc1','op1','oc2','op2',\
        'mp','comm','tgt_p','delta','vega','beup','bedn','strike']
    flask_dt=['entry_dt', 'exp_dt','tgt_dt','earn_dt', 'event_dt', 'div_dt']
          
    for x in flask_val:
        dt[x]=dt[x].replace('',0).astype(float)

    for x in flask_dt:
        for ch in ['','None','N','N/A','0']:
            dt[x]=dt[x].replace(ch,dummy_dt).astype(str)
 #       dt[x]=[parser.parse(i) for i in dt[x]]
        dt[x]=dt[x].apply(lambda i: parser.parse(i))

#existing missing dt     
    dt['si']=dt['si'].str.replace('%','')
    dt['si']=dt['si'].replace('',0).astype(float)
    dt_string=['beta','i_iv','i_rsi','i_rtn_22_pct','i_srtn_22_pct'] 
    for x in [dt_string]:
        dt[x]=dt[x].replace('',0).astype(float)
    
    dt_val=['beta', 'cap', 'entry_p', 'fm_200', 'fm_50', 'fm_hi', 'fm_lo', \
        'hv_22', 'i_hv_22', 'i_iv', 'i_rsi', 'i_rtn_22_pct', 'i_srtn_22_pct', \
        'close', 'rtn_22', 'rtn_22_pct', 'rtn_5_pct',\
       'rtn_66_pct', 'sfm_200', 'sfm_50', 'si', 'spike', 'srtn_22',\
       'srtn_22_pct', 'srtn_66_pct', 'strike', 'vega']
    dt=type_convert(dt, dt_val, 'float')
    
# rgx() should be in unop_mc()
#    def rgx(val):
#        import re
#        evt=re.search('(\d{1,2}-\S{3}-\d{4})',val)
#        if evt:
#            return evt.group(0)
#        else:
#            return np.nan        
#    dt['event_dt']=dt['event'].apply(rgx)    
    
# UPDATE derived: ic, ip, oc, op, lc, lp, mp, risk_orig, risk_live, pl_r, pl_ur, pl_pct, 
    dt['ic']= dt['ic1']+dt['ic2']
    dt['ip']=(dt['ic1']*dt['ip1']+dt['ic2']*dt['ip2'])/dt['ic']    
    dt['oc']= dt['oc1']+dt['oc2']
    #op1, op2 default to be 0
    con_oc=dt['oc']>0
    dt.loc[con_oc, 'op']=(dt['oc1']*dt['op1']+dt['oc2']*dt['op2'])/dt['oc']  
    dt['lc']=dt['ic']- dt['oc']
    #for live trade only
    con_lc=dt['lc']>0
    dt.loc[con_lc, 'lp']= (dt['ic']*dt['ip']-dt['oc']*dt['op'])/dt['lc']
# mp default to be 0 if blank @flask
#    dt[pd.isnull(dt.mp)]['mp']=dt['lp']  
    dt['risk_orig']=dt['ic']*dt['ip']*100+dt['comm']
    dt['risk_live']=dt['lc']*dt['mp']*100
    dt['pl_r']=(dt['op']-dt['ip'])*dt['oc']*100
    dt['pl_ur']=(dt['mp']-dt['lp'])*dt['lc']*100
    dt['pl']=dt['pl_r'] + dt['pl_ur']- dt['comm']
    dt['pl_pct']=(dt['pl']/dt['risk_orig'])
    
# move non-live ticker to tbl_c_hist
#    con_dead=dt['exit_dt']=='N'
#    dt_dead=dt[con_dead]
#    dt_live=dt[~ con_dead]
#    to_sql_delete("delet from tbl_c where exit_dt <>'N'")
    dt['days_to_exp']=dt['exp_dt'].subtract(dt['date']).dt.days   
    dt['days_all']= dt['exp_dt'].subtract(dt['entry_dt']).dt.days   
    dt['days_pct']=(1-dt['days_to_exp']/dt['days_all']).round(2)
    dt['days_to_div']=dt['div_dt'].subtract(dt['date']).dt.days
    dt['days_to_earn']=dt['earn_dt'].subtract(dt['date']).dt.days
    dt['days_to_event']=dt['event_dt'].subtract(dt['date']).dt.days
    
#ALERT data prepare
#    dt['fm_strike']=(dt['close']/dt['strike']-1).round(2)
    dt['srtn_22_chg']=dt['srtn_22_pct']-dt['i_srtn_22_pct']
    dt['rtn_22_chg']=dt['rtn_22_pct']-dt['i_rtn_22_pct']
    dt['hv_22_chg']=dt['hv_22']-dt['i_hv_22']
#    dt['rsi_chg']=dt['rsi']<dt['i_rsi']
    dt['weigh']=dt['risk_live']/dt['cap']
    dt['days_etd']=dt['days_all']-dt['days_to_exp']
    
    dt['sig_etd']=dt['entry_p']*dt['i_iv']/100*np.sqrt(dt['days_etd']/252)
    dt['spike_etd']=(dt['close']-dt['entry_p']/dt['sig_etd'])
    dt['sig_etd2']=dt['entry_p']*dt['i_hv_22']/100*np.sqrt(dt['days_etd']/252)
    dt['spike_etd2']=(dt['close']-dt['entry_p']/dt['sig_etd2'])    
    dt['spike_etd']=dt[['spike_etd','spike_etd2']].max(axis=1)

#Track @ Act & asset level
    z1=dt.groupby('act')['risk_live'].sum()    
    z2=dt.groupby('act')['weigh'].sum()
    z3=dt.groupby('act')['pl'].sum()
    z=list(zip(z1, z2, z3))
    dt_overview=pd.DataFrame(z,columns=['risk_live','weigh', 'pl'], index=z1.index)
    print (" - --Overview ---- \n # of tickers: %s \n"%dt.shape[0], dt_overview)   
    '''
    ALERT
    ? stop: R
    ? exit: Rx
    ? out: tgt_dt min(1/2, cat_dt)
    ? event_dt
    ? unop: mc/bc, spike, hv_22_chg, 
    ? momt: rtn_22_chg, srtn_22_chg, key_level
    ? sizing: weigh
    ? itm/be: 
    '''
## UPDATE Alert - variables, 
    dr=read_sql("select * from tbl_risk")
    R=0.025
    Rx=1.2
#   dt['r']=dt['cap']*r
    stop_pct= -0.3
    exit_pct=0.3
#    p_runaway_pct= 0.1
#    v_stk_vol_pct=2.5
    key_level_pct=0.01    
    exit_days_pct=0.5
    exit_days_to_exp=5
      #net of comm
    event_days=5
    weigh_per_ticker=0.2
    itm_pct=0.02
    spike_std=2
    hv_chg_pct=0.2
    
    CON_stop= (dt['pl_pct']<= stop_pct) |(dt['pl']< -dt['cap']*R)
    CON_out=(dt['tgt_dt']<q_date) | (dt['days_pct']>=exit_days_pct)
    CON_exit=dt['pl_pct']> exit_pct
    
# alert_itm:    
    con_l=dt['lsnv']=='L'
    con_s=dt['lsnv']=='S'
    con_n=dt['lsnv']=='N'
    con_v=dt['lsnv']=='V'
    con_fm_strike_up=(dt['close']/dt['strike']-1)>=itm_pct
    con_fm_strike_dn=(dt['close']/dt['strike']-1)<=(0 - itm_pct)
    
    con_sc_up=(dt['play'].isin(['SC','SCV','SCP','BW'])) & con_fm_strike_up
    con_sp_dn=(dt['play'].isin(['SP','SPV','SCP','SYN'])) & con_fm_strike_dn
    con_nc_up= ((dt['pc']=='C') & (dt['play'].isin(['CAL','BF']))) & con_fm_strike_up
    con_np_dn= ((dt['pc']=='P') & (dt['play'].isin(['CAL','BF']))) & con_fm_strike_up
    CON_itm=con_sc_up | con_sp_dn | con_nc_up |con_np_dn 
#a_be
    CON_be=(dt.close>dt.beup) | (dt.close< dt.bedn)
#a_event        
    con_earn_dt=dt['days_to_earn']>0
    con_div_dt=dt['days_to_div']>0
    con_event_dt=dt['days_to_event']>0
    
    CON_event=(dt['days_to_div']<=event_days & con_div_dt)\
            |(dt['days_to_earn']<=event_days & con_earn_dt)\
              |(dt['days_to_event']<=event_days & con_event_dt )  
#a_momt    
    con_momt_l= con_l & (  (dt['srtn_22_pct']<dt['i_srtn_22_pct']) \
                    | (dt['rtn_22_pct']<dt['i_rtn_22_pct']) )
    con_momt_s= con_s & ( (dt['srtn_22_pct']>dt['i_srtn_22_pct']) \
                   | (dt['rtn_22_pct']>dt['i_rtn_22_pct']) )
    CON_momt= con_momt_l | con_momt_s
#a_key_level
    CON_key_level=(np.abs(dt['fm_50'])<= key_level_pct) \
          |(np.abs(dt['fm_200'])<= key_level_pct) \
          |(np.abs(dt['fm_hi'])<= key_level_pct) \
          |(np.abs(dt['fm_lo'])<= key_level_pct)   
#a_unop              
    CON_unop=pd.notnull(dt['mc']) |pd.notnull(dt['bc'])  
#a_weigh
    CON_weigh=dt['weigh'] >=weigh_per_ticker
#a_vol
    dt['i_hv_22']=1
    play_lv=['SC','SP','SCV','SPV','IC','BW','BF']
    play_hv=['LC','LP','LCV','LPV','LCP','CAL','SYN']
    con_lv=dt.play.isin(play_lv)
    con_hv=dt.play.isin(play_hv)
    con_hv_up=(dt['hv_22']/dt['i_hv_22']> (1+hv_chg_pct)) & con_lv
    con_hv_dn=(dt['hv_22']/dt['i_hv_22']< (1-hv_chg_pct)) & con_hv
    con_spike= (dt['spike']>spike_std) |(dt['spike_etd']>spike_std)
    CON_hv=con_hv_up | con_hv_dn| con_spike
    
#UPDATE alert
#    dt['buff']=1- dt.groupby('act')['risk_live'].sum()/dt['cap']
#    CON_buff=dt['buff']<=buff
    dt['a_out']=CON_out
    dt['a_stop']=CON_stop
    dt['a_exit']=CON_exit
    dt['a_itm']=CON_itm
    dt['a_be']=CON_be
    dt['a_event']=CON_event
    dt['a_momt']=CON_momt 
    dt['a_key']=CON_key_level
    dt['a_unop']=CON_unop
    dt['a_weigh']=CON_weigh
    dt['a_hv']=CON_hv
#    dt['a_buff']=CON_buff

    dt.replace(False, "", inplace=True)
    dt.replace(np.nan, "", inplace=True)
    dt[con_event_dt]['event_dt']=''
    dt[con_earn_dt]['earn_dt']=''    
    dt[con_div_dt]['div_dt']=''
    
    try:
        dt.drop(['index'], axis=1, inplace=True)
    except:
        pass
#APPEND to tbl_track_hist   (record of full trade for Analysis)
#    dt_track_hist=dt[dt.ticker.isin(dt_hist.ticker)]
#    to_sql_append(dt_track_hist, "tbl_track_hist")
#    
##REPLACE tbl_track with latet data    
#    dt_track_live=dt[dt.ticker.isin(dt.ticker)]
#    to_sql_replace(dt_track_live, "tbl_track")
#    df=dt_track_live
    show_alerts=['ticker','risk_live', 'pl_pct', 'pl', 'a_out', 'a_stop', 'a_exit',\
                 'a_itm','a_event', 'a_be','a_momt','a_unop', 'a_hv','a_key', 'a_weigh']
    show_base=['ticker','close','risk_live','pl_pct','lp', 'mp','tgt_p','tgt_dt','days_pct']
    
    show_stop=show_base + ['bedn','beup']
    show_out=show_base
    show_exit=show_base + ['days_to_exp']
    show_itm=show_base +['lsnv','pc','play','strike']
    show_be=show_base+['bedn','beup']
    show_unop=['ticker','mc','bc']
    show_event= show_base + ['div_dt','event_dt', 'earn_dt']
    show_momt=show_base+['sec','rtn_22_chg','srtn_22_chg']
    show_hv=show_base +['spike','spike_etd','hv_22','i_hv_22','i_iv']

    dt.sort_values(['risk_live','pl_pct','days_pct'], ascending=False, inplace=True)
    pd.set_option('display.expand_frame_repr', False)
    print(" \n---- ALERT  ---  \n ", dt[show_alerts])
    print("\n ---- STOP ---   \n ", dt[CON_stop][show_stop])
    print("\n ---- OUT ---    \n ", dt[CON_out][show_out])
    print("\n ---- EXIT ---   \n ", dt[CON_exit][show_exit])
    print("\n ---- HV ---   \n ", dt[CON_hv][show_hv])
    print("\n ---- ITM ---    \n ", dt[CON_itm][show_itm])
    print("\n ---- BE ---   \n ", dt[CON_be][show_be])
    print("\n ---- UNOP ---    \n ", dt[CON_unop][show_unop])
    print("\n ---- EVENT---   \n ", dt[CON_event][show_event])
    print("\n ---- MOMT ---    \n ", dt[CON_momt][show_momt])
 
    pd.set_option('display.expand_frame_repr', True)
    return dt
# INTEL_op

#3. p/c premium sentiment
#4
#Source: tbl_mc_raw, bc_raw ->tbl_spec_candy
def spec_candy_old(q_date, spec_list=[]):
    p_date=q_date-datetime.timedelta(10)
    if len(spec_list)==0:
        print("empty spec_list")
        return
    dmc=read_sql("SELECT * FROM tbl_mc_raw where date>='%s'"%p_date, q_date)
    dbc=read_sql("SELECT * FROM tbl_bc_raw where date>='%s'"%p_date, q_date)
    dmc.drop('index', axis=1, inplace=True)
    dbc.drop(['index','iv','p_chg'], axis=1, inplace=True)
    dmbc=dmc.merge(dbc, on=['ticker','date'],how='outer')
    df=dmbc[dmbc.ticker.isin(spec_list)]
    df.sort_values(by=['ticker','date'], ascending=[True, False], inplace=True)
    df['note']=''
    df['lsn']=''
    df.drop_duplicates('ticker', keep='first',inplace=True)
    df['date']=pd.to_datetime(df['date'])
    to_sql_append(df,"tbl_spec_candy")
    print("tbl_spec_cany appended")
    return df