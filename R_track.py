# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np 
import datetime as datetime
import sqlite3 as db
from timeit import default_timer as timer
from R_plot import plot_base
from dateutil import parser
import scipy.stats

from termcolor import colored, cprint
import warnings
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.2f}'.format
from P_commons import to_sql_append, to_sql_replace, read_sql, type_convert,to_sql_delete
from P_commons import * 
from R_stat import stat_VIEW, stat_run_base

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
    spec_list can have ticker not in tbl_candies
    '''
    lookback=10
    spec_list=[x.upper() for x in spec_list]
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
    
    ds=stat_VIEW(q_date, spec_list, 'update') 
    col_ds=ds.columns.tolist()
    col_df=df.columns.tolist()
    col_nds=list(set(col_df)-set(col_ds))+['ticker']
    df_nds=df[col_nds]
    df=pd.merge(df_nds, ds, how='outer', on='ticker')
    #extra field for flask_candy input & 
    scm_t_spec=read_sql("select * from tbl_spec_candy").columns
    col_extra=list(set(scm_t_spec)-set(df.columns))
    for x in col_extra:
        df[x]=np.nan
    df.drop('index',axis=1, inplace=True)
    scm_t_spec=df.columns.tolist()
    df=df[scm_t_spec]
    #if ticker not in tbl_candies
#    df.loc[pd.isnull(df.date), 'date']=q_date
    df['date']=q_date
    to_sql_append(df,"tbl_spec_candy")
    print("tbl_spec_cany appended  %s: %s "%(q_date, df.ticker.tolist()))
    return df

#Source: spec_candy past 10 days, Dest: replace tbl_trade_candy
def trade_candy(q_date, trade_list=['']):
    '''
    ticker MUST be in tbl_spec_candy
    '''
    #    lookback=30
    trade_list=[x.upper() for x in trade_list]
    #    candy_date=datetime.datetime.today()-datetime.timedelta(lookback)  #whitin 7 days
    df=read_sql("SELECT * FROM tbl_spec_candy", q_date)
    df=df[df.ticker.isin(trade_list)]
    df['date']=pd.to_datetime(df['date'])
    df.sort_values(['ticker','date'], ascending=True, inplace=True)
    df.drop_duplicates('ticker', keep='last',inplace=True)
    if df.empty:
        print ("trade_candy: not in tbl_spec_candy")
        return
    #Update with stat_VIEW @entry date 
    list_trade= df.ticker.tolist()
    ds=stat_VIEW(q_date, list_trade) 
    col_ds=ds.columns.tolist()
    col_df=df.columns.tolist()
    col_nds=list(set(col_df)-set(col_ds))+['ticker']
    df_nds=df[col_nds]
    df=pd.merge(df_nds, ds, how='left', on='ticker')   
    df['date']=q_date
    df.drop('index',axis=1, inplace=True)
    scm_t_spec=df.columns.tolist()
    col_rn=['close','iv','hv_22','rtn_22_pct','srtn_22_pct', 'strike']
    col_rnto=['entry_p','i_iv','i_hv_22','i_rtn_22_pct','i_srtn_22_pct','strik']
    df.rename(columns=dict(zip(col_rn, col_rnto)), inplace=True)
    col_df=df.columns.tolist()
    
    df_trade=read_sql("SELECT * FROM tbl_c")
    df_trade.drop('index', axis=1, inplace=True)
    scm_t_trade=df_trade.columns.tolist()
    
    col_flask=list(set(scm_t_trade)- set(col_df))
    df_extra=pd.DataFrame(columns=col_flask, index=df.index)
    dc=pd.concat([df, df_extra], axis=1)
    scm_t_trade_candy=dc.columns.tolist()
#pre_fill values
    flask_val=['ic1','ip1','ic2','ip2','oc1','op1','oc2','op2',\
        'mp','comm','tgt_p','delta','vega','beup','bedn','strike']
    flask_dt=['entry_dt', 'tgt_dt']
    dc['act']='IBQ'
    dc[flask_val]=0
    dc[flask_dt]=q_date
    dc['exit_dt']='N'
    
    to_sql_replace(dc,"tbl_trade_candy")
    print(" tbl_trade_candy is replaced %s: "%dc.ticker.tolist() )    
    show=['ticker','entry_p', 'earn_dt', 'div_dt', 'i_srtn_22_pct',\
       'i_rtn_22_pct',  'si', 'date','note']
    dc.replace(np.NaN, '', inplace=True)
    print(dc[show])
    return dc

def trade_entry(q_date):
    df=read_sql("SELECT * FROM tbl_trade_candy", q_date)
    dr=read_sql("SELECT * FROM tbl_risk")
# data validation from flask before enter tbl_trade_candy
    df['exit_dt']='N'
    df['lsnv']=df['lsnv'].astype('str').str.upper()
    df['play']=df['play'].astype('str').str.upper()
    df['pc']=df['pc'].astype('str').str.upper()
    con_lsnv= df['lsnv'].isin(['L','S','N','V']).all()
    con_play= df['play'].isin(dr.play.tolist()).all()
    man_val=['comm','delta','i_iv','ic1','ip1','vega','mp','strike']
    
    if  not ( con_lsnv & con_play):
        print("trade_entry: lsnv or play not in list")
        return    
    elif (df[man_val]==0).any(axis=1).values[0]:
        print("trade_entry: mandtory input: %s"%man_val)
        return

        
    ask=input(" --- trade_entry?  yes/n:    ")  
    if ask=='yes':
        scm_t_c=read_sql("select * from tbl_c").columns.tolist()
        scm_t_c=[x for x in scm_t_c if x not in \
            ['index','level_0','cap','i_rsi','i_mc']]
        df=df[scm_t_c]
        to_sql_append(df, 'tbl_c')  
        print (" tbl_c appended: %s"%df.ticker.tolist())
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
    show=['ticker', 'lsn', 'pct_c', 'v_pct','p', 'earn_dt', 'iv',\
        'type', 'strike','vol', 'oexp_dt', 'v_oi', 'bs', 'note']
    df=df[show]
    print("spec_track  %s  days plot "%days_trace)
    plot_base(q_date, df.ticker.unique(),df)
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
    dt_all=read_sql("SELECT * FROM tbl_c", q_date)
 #Validate act, cap   
    dt_all['act']=dt_all['act'].str.upper()
    dt_all.drop('cap', axis=1, inplace=True)
#update cap data
    df_cap=read_sql("SELECT * FROM tbl_act", q_date)
    dt_all=dt_all.merge(df_cap[['act','cap']], on='act',how='left')
# updte with latest stat_VIEW
    ds=stat_VIEW(q_date, dt_all['ticker'].tolist())
    col_share=list(set(dt_all.columns).intersection(set(ds.columns)))
    col_drop=[x for x in col_share if x not in ['ticker']]
    dt_all.drop(col_drop, axis=1, inplace=True)
    dt=pd.merge(dt_all, ds, on='ticker')

# update 'mc/bc" from today unop?
    df_mc=read_sql("SELECT * FROM tbl_mc_raw ", q_date)
    df_mc['date']=pd.to_datetime(df_mc['date'])
    df_mc=df_mc[df_mc.date==q_date]
    loc_mc=dt['ticker'].isin(df_mc['ticker'])
    if loc_mc.any():
        dt.loc[loc_mc,'mc']=df_mc['ticker'] +'|' + df_mc['v_pct'].astype(str)\
            +'|'+df_mc['pct_c'].astype(str)
    else:
        dt['mc']=''
    
    df_bc=read_sql("SELECT * FROM tbl_bc_raw ", q_date)
    df_bc['date']=pd.to_datetime(df_bc['date'])
    df_bc=df_bc[df_bc.date==q_date]
    loc_bc=dt['ticker'].isin(df_bc['ticker'])
    if loc_bc.any():
        dt.loc[loc_bc,'bc']=df_bc['ticker'] +'|' + df_bc['v_oi'].astype(str)\
            +'|'+df_bc['type'].astype(str)
    else:
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
 #force tgt_dt to be    
    dt['tgt_dt']=dt['entry_dt']+(dt['exp_dt']-dt['entry_dt'])/2
    
#existing missing dt     
    try:
        for ch in ['%','None','N','N/A','0']:
            dt['si']=dt['si'].str.replace(ch,'')
        dt['si']=dt['si'].replace('',0).astype(float)
    except:
        pass

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
    dt.loc[~ con_oc, 'op']=0
    dt['lc']=dt['ic']- dt['oc']
    #for live trade only
    if (dt['lc']<0).any():
        print(" track_raw: live contract < 0")
        return
    con_lc=dt['lc']>0
    dt.loc[con_lc, 'lp']= (dt['ic']*dt['ip']-dt['oc']*dt['op'])/dt['lc']
    dt.loc[~con_lc, 'lp']=0
#    dt[pd.isnull(dt.mp)]['mp']=dt['lp']  
    dt['risk_orig']=dt['ic']*dt['ip']*100+dt['comm']
    dt['risk_live']=dt['lc']*dt['mp']*100
    dt['pl_r']=(dt['op']-dt['ip'])*dt['oc']*100
    dt['pl_ur']=(dt['mp']-dt['lp'])*dt['lc']*100  #dead trade->lc=0
    dt['pl']=dt['pl_r'] + dt['pl_ur']- dt['comm']
    dt['pl_pct']=(dt['pl']/dt['risk_orig'])

    dt['days_to_exp']=dt['exp_dt'].subtract(dt['date']).dt.days   
    dt['days_all']= dt['exp_dt'].subtract(dt['entry_dt']).dt.days   
    dt['days_pct']=(1-dt['days_to_exp']/dt['days_all']).round(2)
    dt['days_to_div']=dt['div_dt'].subtract(dt['date']).dt.days
    dt['days_to_earn']=dt['earn_dt'].subtract(dt['date']).dt.days
    dt['days_to_event']=dt['event_dt'].subtract(dt['date']).dt.days
    
    dt['iv_hv']=dt['iv']/dt['hv']
    #get prob_t, prob
    
    dt['days_to_tgt_dt']=dt['tgt_dt'].subtract(dt['entry_dt']).dt.days 
    p_tgt_dt_std=dt['iv']*np.sqrt(dt['days_to_tgt_dt']/252)
    p_std=dt['iv']*np.sqrt(dt['days_to_exp']/252)
    p_mean=dt['close']
    if dt['bedn']==0:  #less or equal
        dt['prob_t']=scipy.stats.norm.cdf(dt['beup'], p_mean, p_tgt_dt_std)
        dt['prob']=scipy.stats.norm.cdf(dt['beup'], p_mean, p_std)
    elif dt['beup']==0:#greater or equal
        dt['prob_t']=scipy.stats.norm.sf(dt['bedn'], p_mean, p_tgt_dt_std)
        dt['prob']=scipy.stats.norm.sf(dt['bedn'], p_mean, p_std)
    elif (dt['bedn']>0 & dt['beup']>0):
        dt['prob']=1- scipy.stats.norm.cdf(dt['bedn'], p_mean, p_tgt_dt_std)- \
                  scipy.stats.norm.sf(dt['beup'], p_mean, p_tgt_dt_std)        
        dt['prob']=1- scipy.stats.norm.cdf(dt['bedn'], p_mean, p_std)- \
                  scipy.stats.norm.sf(dt['beup'], p_mean, p_std)
    else: 
        print("track_raw:  ln319: beup/bedn both 0")
        return
    
#ALERT data prepare
    #    dt['fm_strike']=(dt['close']/dt['strike']-1).round(2)
    dt['weigh']=dt['risk_live']/dt['cap']
    dt['srtn_22_chg']=dt['srtn_22_pct']-dt['i_srtn_22_pct']
    dt['rtn_22_chg']=dt['rtn_22_pct']-dt['i_rtn_22_pct']
    dt['hv_22_chg_pct']=dt['hv_22']/dt['i_hv_22']-1
    #    dt['rsi_chg']=dt['rsi']<dt['i_rsi']
    
    dt['days_etd']=dt['days_all']-dt['days_to_exp']
    
    
    dt['sig_etd']=dt['entry_p']*dt['i_iv']/100*np.sqrt(dt['days_etd']/252)
    dt['spike_etd']=(dt['close']-dt['entry_p']/dt['sig_etd'])
    dt['sig_etd2']=dt['entry_p']*dt['i_hv_22']/100*np.sqrt(dt['days_etd']/252)
    dt['spike_etd2']=(dt['close']-dt['entry_p']/dt['sig_etd2'])    
    dt['spike_etd']=dt[['spike_etd','spike_etd2']].max(axis=1)
    
    #get risk profile (r, rx, risky)
    dr=read_sql("SELECT * FROM tbl_risk")
    dt=pd.merge(dt, dr, on='play')

#no new column from here onward
# keep live trade only -> for proper alert eg. weigh calculation
# move non-live ticker to tbl_c_hist, 
    scm_t_c=[x for x in dt.columns if x not in ['index']]
    dt=dt[scm_t_c]
    con_live=dt['exit_dt']=='N'
    dt_dead=dt[~ con_live]
    try:
        z3_test=dt.groupby('act')['pl'].sum()
    except:
        print("track_raw: ln341:  Must produce aggregated value")
    #validate dead trade
    con_lc=dt['lc']==0  #no live contract
    if (con_live & con_lc).any() | (~con_live & ~con_lc).any():
        print("track_raw: ln337, con_live <>con_lc")
    if ~ dt_dead.empty:
        to_sql_append(dt_dead, 'tbl_c_hist')
        print("track_raw: tbl_c_hist appended: %s"%dt_dead.ticker.tolist())
#        to_sql_delete("delete from tbl_c where exit_dt <>'N'")
    dt=dt[con_live]
    to_sql_replace(dt, 'tbl_c')

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
    Max_loss_per_pos=0.05
    Rx=1.2
#   dt['r']=dt['cap']*r
    stop_pct= -0.3
    exit_pct=0.3
#    p_runaway_pct= 0.1
#    v_stk_vol_pct=2.5
    key_level_pct=0.01    
    exit_days_pct=0.5
    event_days=5
    weigh_per_ticker=0.2
    itm_pct=0.02
    spike_min=2
    hiv_chg_pct=0.2
    hv_rank_chg_min=0.1
    
    CON_stop= (dt['pl_pct']<= dt['r']) 
    CON_out=(dt['tgt_dt']<q_date) | (dt['days_pct']>=exit_days_pct)
    CON_exit=dt['pl_pct']> exit_pct
    
# alert_itm:    
    con_l=dt['lsnv']=='L'
    con_s=dt['lsnv']=='S'
    con_l=dt['detla']>0
    con_s=dt['delta']<0
    
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
#    dt['i_hv_22']=1
    play_lv=['SC','SP','SCV','SPV','IC','BW','BF']
    play_hv=['LC','LP','LCV','LPV','LCP','CAL','SYN']
    con_lv_play=dt.play.isin(play_lv)
    con_hv_play=dt.play.isin(play_hv)
    con_lv_play=dt['vega']<0
    con_hv_play=dt['vega']>0
    #if i_hv_rank exist
    con_hv_up=(dt['hv_22']/dt['i_hv_22']> (1+hiv_chg_pct)) |\
            ((dt['hv_rank']-dt['i_hv_rank'])>hv_rank_chg_min)
    con_iv_up=( dt['iv']/dt['i_iv']>(1+hiv_chg_pct) ) 
  
    con_hv_dn=( dt['hv_22']/dt['i_hv_22']< (1 - hiv_chg_pct) ) |\
            ((dt['hv_rank']-dt['i_hv_rank']) < -hv_rank_chg_min)   
    con_iv_dn=dt['iv']/dt['i_iv']>(1 - hiv_chg_pct)  
    
    con_hiv_up= con_lv_play & (con_hv_up | con_iv_up)
    con_hiv_dn=con_hv_play & (con_hv_dn | con_iv_dn)
    con_hiv= con_hiv_up | con_hiv_dn
    CON_spike= (np.abs(dt['spike'])>spike_min) |(np.abs(dt['spike_etd'])>spike_min) #action on any SPIKE

#a_dir_wrong
    con_dir_ln=(dt['lsnv']=='LN') & (dt['delta']<0)
    con_dir_sn=(dt['lsnv']=='SN') & (dt['delta']>0)
    con_dir_n=(dt['lsnv']=='N') & CON_spike  #main play is N, start trending
    con_dir_v=(dt['lsnv']=='V') & con_hiv  # main play is Vol
    CON_dir=(con_dir_ln | con_dir_sn |con_dir_n | con_dir_v)
#a_size
    CON_size=(dt['var_id'], dt['var_hd']).max(axis=1)*2>( dt['cap']*Max_loss_per_pos)
    
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
    dt['a_dir']=CON_dir
    dt['a_spike']=CON_spike
    dt['a_size']=CON_size

#clean display
    dt.replace(False, "", inplace=True)
    dt.replace(np.nan, "", inplace=True)
    dt.loc[~con_event_dt, 'event_dt']=np.nan
    dt.loc[~con_earn_dt, 'earn_dt']=np.nan    
    dt.loc[~con_div_dt, 'div_dt']=np.nan
    CON_event_show=pd.notnull(dt['div_dt'])|pd.notnull(dt['earn_dt'])\
        |pd.notnull(dt['event_dt'])

    show_alerts=['ticker','risk_live', 'pl_pct', 'pl', 'beta','si','a_out', 'a_stop', 'a_exit',\
                 'a_itm','a_event', 'a_be','a_momt','a_unop', 'a_hv','a_key', 'a_weigh']
    show_base=['ticker','risk_live','pl_pct','days_pct', 'lp', 'mp','tgt_p','close','beta','si']
    
    show_stop=show_base + ['bedn','beup']
    show_out=show_base
    show_exit=show_base + ['days_to_exp']
    show_itm=['ticker','days_pct','close', 'strike', 'lsnv','pc','play']
    show_be=['ticker','days_pct','close','bedn','beup']
    show_unop=['ticker','mc','bc']
    show_event= ['ticker','days_pct','close','div_dt','event_dt', 'earn_dt']
    show_momt=['ticker', 'entry_p','strike','close', 'play','sec','rtn_22_chg','srtn_22_chg']
    show_dir=['ticker','var_id', 'vega','iv','i_iv', 'hv_22','hv_22_chg_pct','iv_hv','play']
    show_spike=['ticker','var_id', 'spike','spike_etd','delta','close','entry_p', 'pl_pct','days_pct']
    
    dt.sort_values(['risk_live','pl_pct','days_pct'], ascending=False, inplace=True)
    pd.set_option('display.expand_frame_repr', False)
    print(" \n---- ALERT  ---  \n ", dt[show_alerts])
    print("\n ---- STOP ---   \n ", dt[CON_stop][show_stop])
    print("\n ---- OUT ---    \n ", dt[CON_out][show_out])
    print("\n ---- EXIT ---   \n ", dt[CON_exit][show_exit])
    print("\n ---- SPIKE actioin ---   \n ", dt[CON_spike][show_spike])
    print("\n ---- LSNV wrong ---   \n ", dt[CON_dir][show_dir])
    print("\n ---- ITM ---    \n ", dt[CON_itm][show_itm])
    print("\n ---- BE ---   \n ", dt[CON_be][show_be])
    print("\n ---- UNOP ---    \n ", dt[CON_unop][show_unop])
    print("\n ---- EVENT---   \n ", dt[CON_event & CON_event_show][show_event])
    print("\n ---- MOMT ---    \n ", dt[CON_momt][show_momt])
 
    pd.set_option('display.expand_frame_repr', True)
    return dt

# INTEL_op
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
import math
def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom