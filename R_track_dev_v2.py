# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 22:28:44 2018

@author: jon
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 20:41:02 2018
@author: jon
"""

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np 
import datetime as datetime
import sqlite3 as db
from timeit import default_timer as timer
#from yahoo_finance import Share
from termcolor import colored, cprint
import warnings
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.2f}'.format

#Import functions
from P_commons import to_sql_append, to_sql_replace, read_sql

capital=0
#DF_sp500=pd.read_csv('c:\\pycode\pyprod\constituents.csv')
#DF_etf=pd.read_csv('c:\\pycode\pyprod\etf.csv')
#df_sp500=DF_sp500.ix[:,0] #serie
#df_etf=DF_etf.ix[:,0]
todate=datetime.date.today()

'''
1. entry to tbl_candy
2. read to csv, update to tbl_trd
3. daily: track_raw
4. daily: track_derived
5. daily: track_alert
6. daily: track_dash
'''

def main_TRACK(q_date):
    print("trade_track started")    
    dt=track_data(q_date)
    if not dt.empty:
        df=track_show(dt)
    print("trade_track is done")

def candy_to_trd(candy_tickers=['GS','SGMO']):
#trd_entry: ticker, sec, div_dt, earn_dt, i_srtn_22__pct, i_rtn_22_pct, i_rsi, entry_p
#cont: i_mc (concat), i_bc, fm_hi, fm_lo, fm_50, fm_200,beta
#add_filed: acct, termc, play (L,S,N), comm, strike, exp_dt, last_p,\
#generic manual entry:    iiv, beup, bedn, delta, vega, note

#input: ic1, ip1, ic2, ip2, oc1, op1, oc2, op2,
    df=read_sql("SELECT * FROM tbl_mc_candy",todate)

    candy_date=datetime.datetime.today()-datetime.timedelta(30)  #whitin 7 days
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
        'tgt_p','tgt_dt', 'entry_dt','exit_dt']
    df_extra=pd.DataFrame(columns=candy_extra, index=df.index)
    dc=pd.concat([df[show_candy], df_extra])
    dc.replace(np.NaN, '', inplace=True)
    dc=dc[dc.ticker!='']
    dc['exit_dt']='N'
    dc.to_csv('c:\\pycode\playlist\candy_%s.csv'%(todate)) 
    return dc


def trade_to_table():
#    df_trade=pd.read_excel(open(r'c:\pycode\playlist\candy.xlsx','rb'))
    df_trade=pd.read_csv(open(r'c:\pycode\playlist\candy.csv'))
    #df_trade=pd.read_csv('G:\Trading\Trade_python\pycode\\trade.xlsx')
    to_sql_replace(df_trade, 'tbl_c')  #log trades and track performance
    print ("trade is saved in tbl_c")

def track_raw(q_date):  # get close at the exact q_date
#--daily update below
# (fm_stat_VIEW):last_p, rsi, strn_22_pct, rtn_22_pct, fm_hi, fm_lo, fm_50, fm_200,
# ic=ic1+ic2, ip=(ic1*ip1+ic2*ip2)/ic
# oc=oc1+oc2, op=(oc1*op1+oc2*op2)/op
#inter: live contract, live cost(price): lc=ic-oc, lp=(ic*ip-op*oc)/lc, mp (m2m con price)
#mp (default=lp), manual update
#realized pl_r=(op-ip)*oc, unrealized: pl_ur=(mp-lp)*lc 
#risk_live=lc*mc, risk_orig=ic*ip
#mc, bc from tbl_mc, bc if any (then concat to i_mc, i_bc)
#alert: a_be(ip/dn), a_event (div, earn), a_stop(pl), a_exit(to_exp,), 
#alert: a_pvom(fm_strike, fm_entry, key_level, vol, opt_bc/mc, iv, rsi, rsi_chg, srtn_22, rtn_22)

    from R_stat import stat_VIEW, stat_run_base
    from T_intel import get_earnings_ec
#    from R_options import get_option_simple
    
    #fmt='{:0.1f}'
#    if underlying=='sp500':
    q_trade_hist="SELECT * FROM tbl_c WHERE exit_dt<>'N'" #get ride of non-live trade
    q_trade="SELECT * FROM tbl_c WHERE exit_dt ='N'"   #get live trade
# save historical trades
    dh=read_sql(q_trade_hist, q_date)

#    if set(['level_0']).issubset(dh.columns):    
#        dh=dh.drop(['level_0'],1)
#    to_sql_append(dh, 'tbl_c_hist')  #save expired trade to tb_trade_hist
    
    dt=read_sql(q_trade, q_date)
    dt=dt.fillna(0)  #fill Nan with 0 for calc

    con_nullticker=pd.isnull(dt.ticker)
    i_nullticker=dt[con_nullticker].index
    dt=dt.drop(i_nullticker)

# UPDATE RAW from stat_view, len(dt)=len(ds) !!!       
    ds=stat_VIEW(q_date, dt['ticker'].tolist())
    
    ds.sort_values(['ticker'], ascending=True)
    dt.sort_values(['ticker'], ascending=True)

    con=ds['ticker'].isin(dt['ticker'])  #obtain index in ds
    
    dt['date']=q_date
#    ds.rename(columns={'close_qdate':'close'}, inplace=True)
#    dt['close']=ds.loc[con, 'close_qdate'].tolist()  #?? FOR all tickers??
#    dt['rtn_22_pct']=ds.loc[con, 'rtn_22_pct'].tolist()
#    dt['srtn_22_pct']=ds.loc[con, 'srtn_22_pct'].tolist()
#    dt['rsi']=ds.loc[con, 'rsi'].tolist()


#SKIP? as stat_VIEW is for all tickers now ????
#    q_price_all="SELECT * FROM tbl_pv_all WHERE date='%s'"%q_date
#    dp=read_sql(q_price_all, q_date)
#    if dp.empty:
#        print ("tb_price_all no data for %s"%q_date)
#        return pd.DataFrame()
#    dp.sort_values(['ticker'], ascending=True)

#    loc_t=pd.isnull(dt.close)  #ticker not in sp or etf
#    loc_s=ds['ticker'].isin(dt.loc[loc_t,'ticker']) 
#raw fileds to update daily
    dt.drop(['fm_50', 'fm_200', 'fm_hi','fm_lo'], axis=1, inplace=True)  #avoid dupe fields with ds
    raw=['ticker','close','fm_hi','fm_lo','fm_50', 'fm_200',\
    'rsi','srtn_22_pct', 'rtn_22_pct']
#         'mc','bc']
    ds.rename(columns={'close_qdate':'close'}, inplace=True)

    dt=dt.merge(ds[raw], on='ticker', how='left')
 
#    dt=get_earnings_ec(dt)

    # next is to update 'mc', 'bc'
    
#    for r in raw:
#        dt.loc[loc_t,'s%'%r]=ds.loc[loc_s,'s%'%r].tolist()
#    dt.loc[loc_t,'close']=dp.loc[loc_p,'close'].tolist()

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
    
# UPDATE derived:   
    dt.replace("",0, inplace=True)
    dt['ic']= dt['ic1']+dt['ic2']
    dt['ip']=(dt['ic1']*dt['ip1']+dt['ic2']*dt['ip2'])/dt['ic']    
    dt['oc']= dt['oc1']+dt['oc2']
    dt['op']=(dt['oc1']*dt['op1']+dt['oc2']*dt['op2'])/dt['oc']  
#ONE TIME    
#    dt['earn_dt']= dt['earn_dt'].values[0].split()[0]
    
    #UPDATE from external
  
    df_mc=read_sql("SELECT * FROM tbl_mc_raw WHERE date='%s'"%q_date, q_date)
    loc_mc=dt['ticker'].isin(df_mc['ticker'])
    if loc_mc.any():
        dt.loc[loc_mc,'mc']=df_mc['ticker'] +'|' + df_mc['v_pct'].astype(str)\
            +'|'+df_mc['pct_c'].astype(str)
    else:
        dt['mc']=''
    dt['bc']=''
    dt['lc']=dt['ic']- dt['oc']
    
    con_null_op=pd.isnull(dt['op']) #if not sold then lp=0 default
    dt.loc[con_null_op, 'op']=0 #dt.loc[con_null_op, 'ip']
    dt['lp']= (dt['ic']*dt['ip']-dt['oc']*dt['op'])/dt['lc']

    dt['mp']=dt['lp']  #default value

    dt['risk_orig']=dt['ic']*dt['ip']*100+dt['comm']
    dt['risk_live']=dt['lc']*dt['mp']*100

    dt['pl_r']=(dt['op']-dt['ip'])*dt['oc']
    dt['pl_ur']=(dt['mp']-dt['lp'])*dt['lc']
    dt['pl_pct']=((dt['pl_r']+dt['pl_ur'])/dt['risk_orig']-1)*100

    dt['days_to_exp']=pd.to_datetime(dt['exp_dt']).subtract(pd.to_datetime(dt['date'])).dt.days   
    dt['days_all']= pd.to_datetime(dt['exp_dt']).subtract(pd.to_datetime(dt['entry_dt'])).dt.days   
    dt['days_pct']=(1-dt['days_to_exp']/dt['days_all']).round(2)
    dt['days_to_div']=pd.to_datetime(dt['div_dt']).subtract(pd.to_datetime(dt['date'])).dt.days
#    dt['earn_dt'].replace([0,np.nan], "1-Jan", inplace=True)
#    dt['earn_dt']=dt['earn_dt'].astype(str)+ "-2018"
    dt['earn_dt']= dt['earn_dt'].values[0].split()[0]
    dt['days_to_earn']=pd.to_datetime(dt['earn_dt']).subtract(pd.to_datetime(dt['date'])).dt.days
    dt['fm_strike']=(dt['last_p']/dt['strike']-1).round(2)
    dt['srtn_chg']=dt['srtn_22_pct']<dt['i_srtn_22_pct']
    dt['rtn_chg']=dt['rtn_22_pct']>dt['i_rtn_22_pct']
    dt['rsi_chg']=dt['rsi']<dt['i_rsi']
##     
## UPDATE ALERT
    stop_loss_pct= -0.5
    p_runaway_pct= 0.1
    v_stk_vol_pct=2.5
    key_level_pct=0.01    
    exit_days_pct=0.7
    exit_days_to_exp=5
    exit_pl_pct=0.5
    event_days=5
    weigh_ticker=0.2
    buff=0.15
    
    con_l=dt['play'].astype('str')=='L'
    con_s=dt['play'].astype('str')=='S'
    con_n=dt['play'].astype('str')=='N'
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
    
    con_exit_no_chance=((dt['pl_pct']<0 )&(dt['days_pct']>=exit_days_pct))
    dt['tgt_dt']=pd.to_datetime(dt['tgt_dt'])

    con_exit_no_time= (q_date>dt['tgt_dt'])|(dt['days_to_exp']<=exit_days_to_exp)
    con_exit_pl= dt['pl_pct']>exit_pl_pct
    CON_exit= con_exit_no_chance |con_exit_no_time |con_exit_pl

    CON_event=(dt['days_to_div']<=event_days)|(dt['days_to_earn']<=event_days)
    CON_runaway= (np.abs(dt['last_p']/dt['strike']-1)>=p_runaway_pct) \
        |(np.abs(dt['last_p']/dt['entry_p']-1)>=p_runaway_pct)
#    con_v=stk vol chg |pd.notnull(dt['mc']) |pd.notnull(dt['mc'])
    CON_ov=pd.notnull(dt['mc']) |pd.notnull(dt['bc'])

#    con_k= any(list(filter(lambda x:np.abs(x)<=k_level_pct, dt[['fm_hi','fm_lo','fm_50','fm_200']]))
#    con_k=any(np.abs(x)<=k_level_pct for x in dt[['fm_hi','fm_lo','fm_50','fm_200']])

#UPDATE alert
    df_cap=read_sql("SELECT * FROM tbl_cap", q_date)

    dt['cap']=df_cap.loc[df_cap['act'].isin(dt['act'])]['cap']
    dt['weigh']=dt['risk_live']/dt['cap']
#    dt['buff']=1- dt.groupby('act')['risk_live'].sum()/dt['cap']
    dt['buff']=0

    
    CON_weigh=dt['weigh'] >=0.2
#    CON_buff=dt['buff']<=buff
    
    dt['a_stop']=CON_stop
    dt['a_be']=CON_be
    dt['a_exit']=CON_exit
    dt['a_key']=CON_key_level
    dt['a_run']=CON_runaway
    dt['a_momt']=CON_momt
    dt['a_event']=CON_event
    dt['a_ov']=CON_ov
    dt['a_weigh']=CON_weigh
#    dt['a_buff']=CON_buff
    
    dt.replace(False, "", inplace=True)
    dt.replace(np.nan, "", inplace=True)
    try:
        dt.drop(['index','Unnamed: 0'], axis=1, inplace=True)
    except:
        pass
    to_sql_replace(dt, "tbl_track")
    
#    to_sql_replace(dt, 'tbl_track')    
    
    show=['act','ticker','last_p','ip', 'lp', 'mp', 'risk_live','pl_pct', 'days_to_exp']
    show_alerts=['a_stop','a_be','a_exit','a_run','a_key', 'a_momt',\
               'a_event','a_ov', 'a_weigh']
    show=show+show_alerts
    show_base=['ticker','last_p','risk_live','pl_pct','days_pct']
    show_stop=show_base + ['bedn','beup','days_to_exp']
    show_exit=show_base + ['tgt_dt', 'days_to_exp']
    show_runaway_key_momt=show_base+['strike','entry_p','fm_50','fm_200','rtn_chg','srtn_chg','rsi_chg']
    show_ov_event=show_base+['div_dt','event','mc','bc']
    
    pd.set_option('display.expand_frame_repr', False)
    print(" ---- stop loss ---       ")
    print(dt[CON_stop][show_stop])
    print("      ---- exit ---           ")
    print(dt[CON_exit][show_exit]) 
    print("          ---- runawy_key_momt ---   ")   
    print(dt[CON_runaway | CON_key_level | CON_momt][show_runaway_key_momt])
    print("                   --- ov_event  -----")
    print(dt[CON_ov |CON_event][show_ov_event])
    pd.set_option('display.expand_frame_repr', True)
    
#    return dt
#    cap=[['IBQ', 70], ['IBE', 2], ['IBT', 1], ['IBR', 1]]
#    df_cap=pd.DataFrame(data=cap, columns=['act','capital'], index=range(len(cap)))
#    dt['weigh']=dt['erisk']/(df_cap.loc[df_cap['act'].isin(dt.act),'capital'])
#    dt['allo']=dt.groupby('act')['erisk']/(cap[dt['act']])
#    CON_weigh=(dt.weigh > 0.2)
##    con_prexit=pd.isnull(dt.alert_prexit) & dt.exist_pct>0.5  # pre-exit order


#    
#    dt.drop(['level_0'], axis=1, inplace=True)

#    #sync with R_play (trade entry)
#    tbl_trade_view=['date','act','ticker','play','edge','expire_date',\
#        'entry_date','con_1','con_p1','estike_1','be_down','be_up',\
#        'comm','exit_target','close_qdate','now_price','delta',\
#        'sec','srtn_22_pct','srtn_22','rtn_22_pct',	'chg_22_66',\
#        'rtn_5_pct',	'beta','iv_30','iv_30_rank','iv_hv','hv_rank',\
#        'si',	'rsi','v_opt_pct','fm_mean20','fm_mean50','fm_mean200',\
#        'earn_date',	'ex_div','estrike_2','exit_date','fm_hi252',\
#        'fm_lo252','hv_22','hv_66','hi_252','lo_252','mean_20',\
#       'mean_50','mean_200','rtn_5','rtn_22','rtn_66','rtn_66_pct',\
#        'sartn_22','p_22_sig','p_66_sig',	'index','con_2','con_p2',\
#        'close_22b',	'close_66b','con_ex1','con_ex2','con_ex_p1','con_ex_p2']
#    to_sql_replace(dt[tbl_trade_view], 'tb_trade')
#
#    risk_weigh=dt[CON_weigh][['act', 'ticker', 'weigh', 'erisk']]
#    risk_stop=dt[CON_stop][['ticker','p_22_sig', 'close_qdate','estike_1', 'be_up', 'be_down']]    
#    risk_exit=dt[CON_exit][['ticker', 'epnl_pct', 'exist_pct']]
#    risk_price=dt[CON_price][['ticker','epnl_pct', 'fm_strike_pct','spike', 'exist_pct']]
#    risk_event=dt[CON_event][['ticker','ex_div', 'earn_date']]
#
#    dt.sort_values(['erisk', 'act'], ascending=[False, True], inplace=True) 
#    show_1=['act', 'ticker','erisk', 'play', 'weigh','epnl', 'epnl_pct'\
#            , 'A_stop','A_exit', 'A_price', 'A_event']
#    show_2=['ticker','rsi_chg', 'spike','fm_strike_pct','ts_chg', 'sm_chg']
#    
#    print(dt[show_1])
#    print(dt[show_2])
#    print ('risk_weigh')    
#    print(risk_weigh)
#    print(" ------ ")
#    print ('risk_price')    
#    print(risk_price)
#    print(" -------")
#    print (colored('stop loss: HEDGED SURE PROFIT BY EXPIRE???', 'red','on_cyan'))    
#    print(risk_stop)
#    print(" ------")
#    print ('profit exit')    
#    print(risk_exit)
#    print("  -----    ")
##    print (colored('need prexit order', 'red', 'on_cyan'))    
##    print(colored(risk_prexit,'blue'))
#    print("  -----    ")
#    print (colored('event risk', 'magenta', 'on_cyan'))    
#    print(colored(risk_event,'magenta'))
#    print("")
    #print (colored('>>>>>>>>>     ALERT!!!    <<<<<<<<', 'red'))
    #print (df_alert[['ticker', 'alert_stop', 'alert_exit','erisk']])

#    try:
#        SPY=web.get_quote_yahoo('SPY')['last']
#    except:
#        print("web.get_quote_yahoo error@line 289")
#        SPY=1000
#    print (colored('>>>> SPY equivalent number of share in movement <<<<<<<<', 'red'))
#    print("SPY equivalent share in movement:", (dt['beta']*dt['delta']*dt['close_last']).sum()/SPY)



def track_show(dt ):  #risk monitor parameters
    dt.replace("",0, inplace=True)
    dt['contracts']=dt['con_1']+dt['con_2']-dt['con_ex1']-dt['con_ex2']
    dt['entry_price']=(dt['con_1']*dt['con_p1']+dt['con_2']*dt['con_p2'])/(dt['con_1']+dt['con_2'])
    
    con_sold= (dt['con_ex1']+dt['con_ex2'])>0
    dt.loc[con_sold, 'exit_price']=(dt['con_ex1']*dt['con_ex_p1']+dt['con_ex2']*dt['con_ex_p2'])/(dt['con_ex1']+dt['con_ex2'])
    con_m2m=pd.notnull(dt.now_price)  #current m2m price
    dt.loc[con_m2m,'erisk']= dt['contracts']*dt['now_price']*100
    dt.loc[~con_m2m, 'erisk']=dt['contracts']*dt['entry_price']*100  #update exited trade before removal
    dt['epnl']=dt['contracts']*(dt['now_price']-dt['entry_price'])*100-dt['comm']  #unrealized pnl
    dt['epnl_pct']=(dt['now_price']/dt['entry_price']-1)
    dt['days_to_expire']=pd.to_datetime(dt['expire_date'])-pd.to_datetime(dt['date']) 
    dt['days_all_life']= pd.to_datetime(dt['expire_date'])-pd.to_datetime(dt['entry_date'])
    dt['days_from_entry']=dt['days_all_life'] - dt['days_to_expire']
    dt['exist_pct']=1-dt['days_to_expire']/dt['days_all_life']
    dt['exist_pct']=dt['exist_pct'].round(2)
    dt['days_left']=dt['days_to_expire']
    dt['days_to_exdiv']=pd.to_datetime(dt['ex_div']) - pd.to_datetime(dt['date'])
    dt['earn_date'].replace([0,np.nan], "1-Jan", inplace=True)
    dt['earn_date']=dt['earn_date'].astype(str)+ "-2018"
    dt['days_to_earn']=pd.to_datetime(dt['earn_date']) - pd.to_datetime(dt['date'])
    cap=[['IBQ', 5], ['IBE', 2], ['IBT', 1], ['IBR', 1]]
    df_cap=pd.DataFrame(data=cap, columns=['act','capital'], index=range(len(cap)))
    dt['weigh']=dt['erisk']/(df_cap.loc[df_cap['act'].isin(dt.act),'capital'])
#    dt['allo']=dt.groupby('act')['erisk']/(cap[dt['act']])

# risk_price_action    
    dt['fm_strike_pct']=(dt['close_qdate']/dt['estike_1']-1).round(2)
    dt['near_ma']= (np.abs((dt['close_qdate']/dt['mean_50']-1)<=0.01)) \
                    |(np.abs((dt['close_qdate']/dt['mean_200']-1)<=0.01))
    
# risk_price action (rsi, near ma, from strike )
    dt.loc[((dt['play']=='L')|(dt['play']=='LL')), 'alert_rsi']=dt['rsi']<dt['rsi_prev'].astype(float) #less momentum
    dt.loc[((dt['play']=='S')|(dt['play']=='SS')), 'alert_rsi']=dt['rsi']>dt['rsi_prev'].astype(float)
#    dt.loc[(dt['play']=='Z'), 'alert_direction']=\
 #           np.abs(dt['pct_estrike_1'])>np.abs(dt['pct_estrike_prev'])
    CON_price= (np.abs(dt['fm_strike_pct'])>=0.05) | dt['near_ma']| dt['alert_rsi']
    con_too_late= (dt.exist_pct>=0.5) & (dt['now_price']/dt['entry_price']<=1.1)
    con_stop_loss= (dt.epnl_pct<= -0.5) # loss of 30%
    con_be= (dt.close_qdate > dt.be_up.astype(float)) | (dt.close_qdate< dt.be_down.astype(float))
    CON_stop=con_too_late| con_stop_loss | con_be # | con_lastweek
#    con_exit=(dt['play'].str.strip().str[0]=='Z')& (dt.pct_estrike_1 <=0.02) &(dt.exist_pct>=0.5)
    CON_exit=(dt['epnl_pct'] >= 0.3) &(dt.exist_pct>=0.5)
    CON_event= ((dt.days_to_exdiv<='5 days') & (dt.days_to_exdiv>='0 days')) | \
            ((dt.days_to_earn <='5 days') & (dt.days_to_earn>='0 days'))
    CON_weigh=(dt.weigh > 0.2)
#    con_prexit=pd.isnull(dt.alert_prexit) & dt.exist_pct>0.5  # pre-exit order
    dt['A_exit']= CON_exit
    dt['A_price']= CON_price
    dt['A_stop']= CON_stop
    dt['A_event'] = CON_event
    dt['A_weigh'] = CON_weigh
    
    dt.replace(False, "", inplace=True)
    dt.replace(np.nan, "", inplace=True)
    
    dt.drop(['level_0'], axis=1, inplace=True)
    to_sql_replace(dt, 'tb_trade_track')
    #sync with R_play (trade entry)
    tbl_trade_view=['date','act','ticker','play','edge','expire_date',\
        'entry_date','con_1','con_p1','estike_1','be_down','be_up',\
        'comm','exit_target','close_qdate','now_price','delta',\
        'sec','srtn_22_pct','srtn_22','rtn_22_pct',	'chg_22_66',\
        'rtn_5_pct',	'beta','iv_30','iv_30_rank','iv_hv','hv_rank',\
        'si',	'rsi','v_opt_pct','fm_mean20','fm_mean50','fm_mean200',\
        'earn_date',	'ex_div','estrike_2','exit_date','fm_hi252',\
        'fm_lo252','hv_22','hv_66','hi_252','lo_252','mean_20',\
       'mean_50','mean_200','rtn_5','rtn_22','rtn_66','rtn_66_pct',\
        'sartn_22','p_22_sig','p_66_sig',	'index','con_2','con_p2',\
        'close_22b',	'close_66b','con_ex1','con_ex2','con_ex_p1','con_ex_p2']
    to_sql_replace(dt[tbl_trade_view], 'tb_trade')

    risk_weigh=dt[CON_weigh][['act', 'ticker', 'weigh', 'erisk']]
    risk_stop=dt[CON_stop][['ticker','p_22_sig', 'close_qdate','estike_1', 'be_up', 'be_down']]    
    risk_exit=dt[CON_exit][['ticker', 'epnl_pct', 'exist_pct']]
    risk_price=dt[CON_price][['ticker','epnl_pct', 'fm_strike_pct','spike', 'exist_pct']]
    risk_event=dt[CON_event][['ticker','ex_div', 'earn_date']]

    dt.sort_values(['erisk', 'act'], ascending=[False, True], inplace=True) 
    show_1=['act', 'ticker','erisk', 'play', 'weigh','epnl', 'epnl_pct'\
            , 'A_stop','A_exit', 'A_price', 'A_event']
    show_2=['ticker','rsi_chg', 'spike','fm_strike_pct','ts_chg', 'sm_chg']
    
    print(dt[show_1])
    print(dt[show_2])
    print ('risk_weigh')    
    print(risk_weigh)
    print(" ------ ")
    print ('risk_price')    
    print(risk_price)
    print(" -------")
    print (colored('stop loss: HEDGED SURE PROFIT BY EXPIRE???', 'red','on_cyan'))    
    print(risk_stop)
    print(" ------")
    print ('profit exit')    
    print(risk_exit)
    print("  -----    ")
#    print (colored('need prexit order', 'red', 'on_cyan'))    
#    print(colored(risk_prexit,'blue'))
    print("  -----    ")
    print (colored('event risk', 'magenta', 'on_cyan'))    
    print(colored(risk_event,'magenta'))
    print("")
    #print (colored('>>>>>>>>>     ALERT!!!    <<<<<<<<', 'red'))
    #print (df_alert[['ticker', 'alert_stop', 'alert_exit','erisk']])

#    try:
#        SPY=web.get_quote_yahoo('SPY')['last']
#    except:
#        print("web.get_quote_yahoo error@line 289")
#        SPY=1000
#    print (colored('>>>> SPY equivalent number of share in movement <<<<<<<<', 'red'))
#    print("SPY equivalent share in movement:", (dt['beta']*dt['delta']*dt['close_last']).sum()/SPY)




   
    

    


### t_trade
#0. dmc_combo ((update logic & testing)
#
#1. unop_combo ()-> tb_dmc_candy
#2. candy + generic field -> candy.csv (tbl_candy with date)
#3. t_entry: select fm tbl_candy, add generic field -> trd.csv 
#4. manualy input trd.csv -> tbl_trd
#5. track
#5a. update raw (stat_VIEW)
#5b. update derived
#5c. update con
#5d. alert output
#
#
#NEW IDEA
#*get_opt_simple (try brakdown page only to get iv_rank
#2. page_brkdn: dnld one year iv/ vwop data -> plot or ai