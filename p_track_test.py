# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np 
import datetime as datetime
import scipy.stats as stats
import sqlite3 as db
from timeit import default_timer as timer
#from yahoo_finance import Share
import pandas_datareader.data as web
from termcolor import colored, cprint
import warnings
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.2f}'.format

#Import functions
from P_commons import to_sql_append, to_sql_replace, read_sql
from P_intel import get_rsi, get_share_nasdaq, get_tsm
from P_fwdtest import win_loss
from P_options import get_option
from P_intel import marketbeat, get_ta, get_earning_date

capital=20000
DF_sp500=pd.read_csv('c:\\pycode\pyprod\constituents.csv')
DF_etf=pd.read_csv('c:\\pycode\pyprod\etf.csv')
df_sp500=DF_sp500.ix[:,0] #serie
df_etf=DF_etf.ix[:,0]
todate=datetime.date.today()

def trade_track(underlying, reval_date):
# 1. tbl_trade is to record entry of new trades
#save historical trade to tbl_trade_hist
# tbl_trade mandatory: rsi, momt_n, trend_n, etike_1, play (CAL,L,S)
#?? ex_div, p_22_sig?
#! win_loss to think later,
    start_time=timer()
    do=pd.DataFrame()
    if underlying=='sp500':
        q_trade_hist="SELECT * FROM tbl_trade WHERE exit_date<>'N'" #get ride of non-live trade
        q_trade="SELECT * FROM tbl_trade WHERE exit_date ='N'"   #get live trade
    elif underlying=='etf':
        q_trade_hist="SELECT * FROM tbl_trade_etf WHERE exit_date<>'N'" 
        q_trade="SELECT * FROM tbl_trade_etf WHERE exit_date ='N'"
    else:
        print ("trade_track() missing underlying")
        exit
# save historical trades
    dh=read_sql(q_trade_hist, reval_date)
    if set(['level_0']).issubset(dh.columns):    
        dh=dh.drop(['level_0'],1)
    if underlying =='sp500':
        to_sql_append(dh, 'tbl_trade_hist')
    elif underlying == 'etf':
        to_sql_append(dh, 'tbl_trade_hist_etf')
 #   q_trade="SELECT * FROM tbl_trade WHERE exit_date ='N'" #get only live trade
    dt=read_sql(q_trade, reval_date)
    dt=dt.fillna(0)  #fill Nan with 0 for calc
    #get late price for above tickers from tbl_price
    #get previous price pct
#get win_10 to alert for exit
    con_nullticker=pd.isnull(dt.ticker)
    i_nullticker=dt[con_nullticker].index
    dt=dt.drop(i_nullticker)
    dt['pct_estrike_prev']=dt['close_last']/dt['estike_1']-1
    dt['rsi_prev']=dt['rsi']
   #update latest price from tbl_price and rsi

#enrich si_data & earning data

    dt=get_sis_sqz(dt)  #[si, si_chg, pe, hi_1y_fm, lo_1y_fm, ma_200_fm, ma_50_fm, sec, ind]
    dt=get_earnings_ec(dt) #['earn','earn_time','st','it','lt','i_sent','a_sent','com']
    
    
    for index, row in dt.iterrows():
        ticker=row['ticker']
        if underlying=='sp500':
            q_price="SELECT date, %s FROM tbl_price WHERE instr(date,'%s')>0"%(ticker, reval_date)
            q_stat="SELECT * FROM tbl_stat WHERE ticker ='%s'"%ticker
        else: 
            q_price="SELECT date, %s FROM tbl_price_etf WHERE instr(date,'%s')>0"%(ticker, reval_date)
            q_stat="SELECT corr FROM tbl_stat_etf WHERE ticker ='%s'"%ticker
        dp=read_sql(q_price, reval_date)
        dc=read_sql(q_stat, reval_date)
        if not dp.empty:
            close=dp.iloc[0,1]
        else:
            print("no latest price:", ticker)
#            close=dt.loc[index, 'close_last']  #copy from last price
            pass           
        dt.loc[index, 'close_last']= close #update close_last price
        dt.loc[index,'rsi']=float(get_rsi(ticker)) ## Convert to Flaat from Object
#        dt.loc[index,'corr_new']=dc['corr'][0] #dc.iloc[0,0]  ## update latest corr
        dt.loc[index, 'momt_n_last']=dc['momt_n'][0]
        dt.loc[index, 'trend_n_last']=dc['trend_n'][0]
        if ticker =='TLT':
            dt.loc[index, 'beta']=-0.27
        else:    
            dt.loc[index, 'beta']=DF_sp500[DF_sp500['SYMBOL']==ticker].beta.values[0] #series   
#        try:
#            x_si=web.get_quote_yahoo(ticker)['short_ratio'].values
#        
#            x_p_pct=web.get_quote_yahoo(ticker)['change_pct'].values
#        except:
#            x_si=[0]
#            x_p_pct=[0]
        t=[]
        t.append(ticker)
        x_vol, x_vol_avg, x_ex_div, x_beta=1,1,1,1    
        try:
            x_vol, x_vol_avg, x_ex_div, x_beta= get_share_nasdaq(t)
        except:
#            print('get_share_nasdaq error:', ticker)
            pass
        try:
            y_ex_div, dummy_beta=get_share_nasdaq(ticker)  #ex_div from Nasdaq source
            if len(y_ex_div)>0 and (y_ex_div[0] != 'N/A') and (y_ex_div[0] >= x_ex_div):
                dt.loc[index, 'ex_div']=y_ex_div[0] #Nasdq
            else:
                dt.loc[index, 'ex_div']=x_ex_div  #yahoo
        except:
            pass
#        dt.loc[index, 'si']=x_si[0]
#        dt.loc[index, 'p_pct']=x_p_pct[0]
        dt.loc[index, 'v_pct']=float(x_vol)/float(x_vol_avg) #string to float
        dt.loc[index, 'ta']=''        
        try:
            dt.loc[index, 'ta']=get_ta(ticker)
        except:
            pass
# add bechmark       
        if underlying =='sp500':
           try:
               x,y,z=get_tsm(ticker)
           except:
               print(colored("tsm error:  %s"%ticker, 'red', 'on_cyan'))
               x,y,z=0,0,0
           dt.loc[index,'sec']=x
           dt.loc[index, 'rrtn_22_ts']=y
           dt.loc[index,'rrtn_22_sm'] = z
#                print("benchmark error:   ", row['ticker'])
#                pass
        elif underlying =='etf':
            dt.loc[index,'sec']=''
            dt.loc[index,'rrtn_22_ts']=''
            dt.loc[index,'rrtn_22_sm']=''
#        try:
#            do_tmp,  posts=get_option(ticker)
#            do=do.append(do_tmp)
#
#        except:
#            pass

##! Monitor risk criteria
    dt.date_last=reval_date  
    dt['contracts']=dt['con_1']+dt['con_2']-dt['con_ex1']-dt['con_ex2']
    dt['entry_price']=(dt['con_1']*dt['con_p1']+dt['con_2']*dt['con_p2'])/(dt['con_1']+dt['con_2'])
    dt['exit_price']=(dt['con_ex1']*dt['con_ex_p1']+dt['con_ex2']*dt['con_ex_p2'])/(dt['con_ex1']+dt['con_ex2'])
    con_m2m=pd.notnull(dt.exit_target)
    dt.loc[con_m2m,'erisk']= dt['contracts']*dt['exit_target']*100
    dt.loc[~con_m2m, 'erisk']=dt['contracts']*dt['entry_price']*100  #update exited trade before removal

#    dt['epnl']=(dt['con_ex1']+dt['con_ex2'])*(dt['exit_price']-dt['entry_price'])*100-dt['comm']
#    dt['epnl_pct']=0.01*dt['epnl']/((dt['con_1']+dt['con_2'])*dt['entry_price'])
    dt['epnl']=dt['contracts']*(dt['exit_target']-dt['entry_price'])*100-dt['comm']  #unrealized pnl
    dt['epnl_pct']=(dt['exit_target']/dt['entry_price']-1)
    dt['days_to_expire']=pd.to_datetime(dt['expire_date'])-pd.to_datetime(dt['date_last']) 
    dt['days_all_life']= pd.to_datetime(dt['expire_date'])-pd.to_datetime(dt['entry_date'])
# evaluate p_fwd_10 early sign    
    dt['days_from_entry']=dt['days_all_life'] - dt['days_to_expire']
    
# early indicator to stay or exit trade  !!
    dt.loc[dt['days_from_entry']=='5 days','p_5_fwd']=dt['close_last']
    dt.loc[dt['days_from_entry']=='10 days','p_10_fwd']=dt['close_last']
#        dt['p_fwd_10']=dt['close_last']
#    dt=win_loss(dt, 10)  #
    return dt
#    exit
    if underlying=='sp500': 
        con_datetime=(type(dt['ex_div'])==datetime.datetime)
        dt.loc[~con_datetime, 'ex_div']=datetime.datetime(2016,1,1) #day of month out of range error
    else: 
        dt['ex_div']=datetime.datetime(2016,1,1)
    dt['days_to_exdiv']=pd.to_datetime(dt['ex_div']) - pd.to_datetime(dt['date_last'])
    dt['exist_pct']=1-dt['days_to_expire']/dt['days_all_life']
    dt['exist_pct']=dt['exist_pct'].round(2)
    dt['days_left']=dt['days_to_expire']
    dt['days_to_exdiv']=dt['days_to_exdiv']
    #dt calculated fields
    dt['weigh']=dt['erisk']/(dt['erisk'].sum())
    #dt['pct_estrike_1']="{:.1%}".format(dt['close_last'])/dt['estike_1']-1)
    dt['pct_estrike_1']=(dt['close_last']/dt['estike_1']-1).round(2)
    dt['momt_n_chg']=dt['momt_n_last']-dt['momt_n']
    dt['trend_n_chg']=dt['trend_n_last']-dt['trend_n']
    # exit profit trade in time
    con_exit=(dt['play'].str.strip().str[0]=='Z')& (dt.pct_estrike_1 <=0.02) &(dt.exist_pct>=0.5)
# alert on margin of safety  
    con_sig=(dt.close_last > dt.be_up) | (dt.close_last< dt.be_down) | (np.abs(dt.close_last - dt.close_qdate) >=dt.p_22_sig)
#stop loss    
    con_too_late=(dt.exist_pct>=0.5) & (dt['exit_target']/dt['entry_price']<=1.1)
    con_stop_loss=dt.epnl_pct<= -0.35 # loss of 30%
    #con_lastweek=dt.days_left<=4
    con_stop=con_too_late| con_stop_loss# | con_lastweek
    con_div=(dt.days_to_exdiv<='5 days') & (dt.days_to_exdiv>='0 days')
    con_risk_allo=(dt.erisk/capital >0.15) | (dt.weigh > 0.35)
    con_prexit=pd.isnull(dt.alert_prexit) & dt.exist_pct>0.5  # pre-exit order
    
    dt['alert_exit']=con_exit
    dt['alert_sig']=con_sig
    dt['alert_stop']=con_stop
    dt['alert_div']=con_div
    
    dt['move_sig']=(dt['close_last']-dt['close_qdate'])/dt['p_22_sig']    

#Monitor: allocation, break_even, profit_exit, stop_loss, assign_risk,event_risk, assumption wrong: direction, time
    
    risk_allo=dt[con_risk_allo][['ticker', 'weigh', 'erisk']]
    risk_sig=dt[con_sig][['ticker','p_22_sig', 'move_sig','close_last', 'close_qdate','estike_1', 'be_up', 'be_down']]    
    risk_stop=dt[con_stop][['ticker', 'epnl_pct', 'exist_pct']]
    risk_exit=dt[con_exit][['ticker','epnl_pct', 'pct_estrike_1','exist_pct']]
    risk_div=dt[con_div][['ticker','ex_div']]
    risk_prexit=dt[con_prexit][['ticker','epnl_pct', 'exist_pct', 'days_to_expire']]
    dt.alert_stop[dt.alert_stop==False]=""
    dt.alert_exit[dt.alert_exit==False]=""
    dt.alert_div[dt.alert_div==False]=""  
    dt.alert_sig[dt.alert_sig==False]=""  
#    dt.win_5[dt.win_5==False]="" 
#    dt.win_10[dt.win_10==False]="" 
   # dt.wrong[dt.wrong==False]="" 
    #multiple conditional alert
    dt.loc[((dt['play']=='L')|(dt['play']=='LL')), 'alert_direction']=dt['rsi']<dt['rsi_prev'] #less momentum
    dt.loc[((dt['play']=='S')|(dt['play']=='SS')), 'alert_direction']=dt['rsi']>dt['rsi_prev']
    dt.loc[((dt['play']=='CAL')|(dt['play']=='ZZ')|(dt['play']=='Z0')|(dt['play']=='Z4')), 'alert_direction']=\
            np.abs(dt['pct_estrike_1'])>np.abs(dt['pct_estrike_prev'])
    dt.loc[dt['play']=='E', 'alert_direction']=np.abs(dt['pct_estrike_1'])<np.abs(dt['pct_estrike_prev'])
   #counter
#    if underlying=='sp500':
#        dt.loc[dt['alert_stop']==True, 'counter']=dt['counter'].astype(float)+1
#    else:
#        dt['counter']=10
    dt.loc[dt['alert_stop']==True, 'counter']=dt['counter'].astype(float)+1
    dt.alert_direction[dt.alert_direction==False]="" 
    #dt.counter[dt.counter==0]="" 
#    if underlying=='sp500':    
#        dt=dt.drop(['level_0'],1)
#    else:
#        pass
    dt=dt.drop(['level_0'],1)
    if underlying=='sp500':
        to_sql_replace(dt, 'tbl_trade_track')
        to_sql_replace(dt, 'tbl_trade')
    else:
        to_sql_replace(dt, 'tbl_trade_etf')
    #pd.set_option('display.max_columns', None)
    dt['rsi_chg']=dt['rsi']-dt['rsi_prev']
  #  print (colored('>>>>>>>>>     NO exit order!!!    <<<<<<<<', 'red', attrs=['bold', 'reverse','blink']))
    print (colored('allocation risk', 'magenta', 'on_cyan'))    
    print(colored(risk_allo,'magenta'))
    print(" ------ ")
    print (colored('std deviation risk', 'red', 'on_cyan'))    
    print(colored(risk_sig,'blue'))
    print(" -------")
    print (colored('stop loss: HEDGED SURE PROFIT BY EXPIRE???', 'red','on_cyan'))    
    print(colored(risk_stop,'red'))
    print(" ------")
    print (colored('profit exit', 'red', 'on_green'))    
    print(colored(risk_exit,'green'))
    print("  -----    ")
    print (colored('need prexit order', 'red', 'on_cyan'))    
    print(colored(risk_prexit,'blue'))
    print("  -----    ")
    print (colored('event risk', 'magenta', 'on_cyan'))    
    print(colored(risk_div,'magenta'))
    print("")
    #print (colored('>>>>>>>>>     ALERT!!!    <<<<<<<<', 'red'))
    #print (df_alert[['ticker', 'alert_stop', 'alert_exit','erisk']])
    dtt=dt[['ticker','erisk', 'play', 'weigh', 'delta','beta', 'epnl', 'epnl_pct', 'alert_stop','alert_sig', 'alert_exit', 'alert_div', 'days_left',\
    #'alert_direction',\
    'pct_estrike_1','rsi_chg', 'p_pct', 'v_pct', 'si', 'ex_div', 'ta',  \
    'momt_n_chg', 'trend_n_chg', 'win_5', 'win_10', 'sec', 'rrtn_22_ts', 'rrtn_22_sm']]
    dtt.sort_values(['erisk', 'play'], ascending=[False, True], inplace=True)
    try:
        SPY=web.get_quote_yahoo('SPY')['last']
    except:
        print("web.get_quote_yahoo error@line 289")
        SPY=1000
    print (colored('>>>> SPY equivalent number of share in movement <<<<<<<<', 'red'))
    print("SPY equivalent share in movement:", (dt['beta']*dt['delta']*dt['close_last']).sum()/SPY)
    end_time=timer()
    test_time=end_time - start_time
    print("trade_track is done in seconds:  %s"%test_time)
    return dtt


#monitor live trade risks
def trade_track_new(underlying, reval_date):
    start_time=timer()
    df=track_data_prep()
    df=track_criteria(df)
    track_show(df)
    end_time=timer()
    test_time=end_time - start_time
    print("trade_track is done in seconds:  %s"%test_time)

def track_data_prep(underlying, reval_date):
    
    if underlying=='sp500':
        q_trade_hist="SELECT * FROM tbl_trade WHERE exit_date<>'N'" #get ride of non-live trade
        q_trade="SELECT * FROM tbl_trade WHERE exit_date ='N'"   #get live trade
    elif underlying=='etf':
        q_trade_hist="SELECT * FROM tbl_trade_etf WHERE exit_date<>'N'" 
        q_trade="SELECT * FROM tbl_trade_etf WHERE exit_date ='N'"
    else:
        print ("trade_track() missing underlying")
        exit
# save historical trades
    dh=read_sql(q_trade_hist, reval_date)
    if set(['level_0']).issubset(dh.columns):    
        dh=dh.drop(['level_0'],1)
    if underlying =='sp500':
        to_sql_append(dh, 'tbl_trade_hist')
    elif underlying == 'etf':
        to_sql_append(dh, 'tbl_trade_hist_etf')
 #   q_trade="SELECT * FROM tbl_trade WHERE exit_date ='N'" #get only live trade
    dt=read_sql(q_trade, reval_date)
    dt=dt.fillna(0)  #fill Nan with 0 for calc

#get win_10 to alert for exit
    con_nullticker=pd.isnull(dt.ticker)
    i_nullticker=dt[con_nullticker].index
    dt=dt.drop(i_nullticker)
    dt['pct_estrike_prev']=dt['close_last']/dt['estike_1']-1
    dt['rsi_prev']=dt['rsi']
    dt=get_sis_sqz(dt)  #[si, si_chg, pe, hi_1y_fm, lo_1y_fm, ma_200_fm, ma_50_fm, sec, ind]
    dt=get_earnings_ec(dt) #['earn','earn_time','st','it','lt','i_sent','a_sent','com']
    
    
    for index, row in dt.iterrows():
        ticker=row['ticker']
        if underlying=='sp500':
            q_price="SELECT date, %s FROM tbl_price WHERE instr(date,'%s')>0"%(ticker, reval_date)
            q_stat="SELECT * FROM tbl_stat WHERE ticker ='%s'"%ticker
        else: 
            q_price="SELECT date, %s FROM tbl_price_etf WHERE instr(date,'%s')>0"%(ticker, reval_date)
            q_stat="SELECT corr FROM tbl_stat_etf WHERE ticker ='%s'"%ticker
        dp=read_sql(q_price, reval_date)
        dc=read_sql(q_stat, reval_date)
        if not dp.empty:
            close=dp.iloc[0,1]
        else:
            print("no latest price:", ticker)
#            close=dt.loc[index, 'close_last']  #copy from last price
            pass           
        dt.loc[index, 'close_last']= close          #update close_last price
        dt.loc[index,'rsi']=float(get_rsi(ticker))  # Convert to Flaat from Object
        dt.loc[index, 'momt_n_last']=dc['momt_n'][0]    #momn_n
        dt.loc[index, 'trend_n_last']=dc['trend_n'][0]  # trend_n
        if ticker =='TLT':
            dt.loc[index, 'beta']=-0.27             # beta
        else:    
            dt.loc[index, 'beta']=DF_sp500[DF_sp500['SYMBOL']==ticker].beta.values[0] #series   
#        try:
#            x_si=web.get_quote_yahoo(ticker)['short_ratio'].values
#        
#            x_p_pct=web.get_quote_yahoo(ticker)['change_pct'].values
#        except:
#            x_si=[0]
#            x_p_pct=[0]
        t=[]
        t.append(ticker)
        x_vol, x_vol_avg, x_ex_div, x_beta=1,1,1,1    
        try:
            y_ex_div, dummy_beta=get_share_nasdaq(ticker)  #ex_div from Nasdaq source
            if len(y_ex_div)>0 and (y_ex_div[0] != 'N/A') and (y_ex_div[0] >= x_ex_div):
                dt.loc[index, 'ex_div']=y_ex_div[0] #Nasdq
            else:
                dt.loc[index, 'ex_div']=x_ex_div  #yahoo
        except:
            pass
#        dt.loc[index, 'si']=x_si[0]
#        dt.loc[index, 'p_pct']=x_p_pct[0]
#        dt.loc[index, 'v_pct']=float(x_vol)/float(x_vol_avg) #string to float
        dt.loc[index, 'ta']=''        
        try:
            dt.loc[index, 'ta']=get_ta(ticker)
        except:
            pass
# add bechmark       
        if underlying =='sp500':
           try:
               x,y,z=get_tsm(ticker)
           except:
               print(colored("tsm error:  %s"%ticker, 'red', 'on_cyan'))
               x,y,z=0,0,0
           dt.loc[index,'sec']=x
           dt.loc[index, 'rrtn_22_ts']=y
           dt.loc[index,'rrtn_22_sm'] = z
#                print("benchmark error:   ", row['ticker'])
#                pass
        elif underlying =='etf':
            dt.loc[index,'sec']=''
            dt.loc[index,'rrtn_22_ts']=''
            dt.loc[index,'rrtn_22_sm']=''
    dt.date_last=reval_date  
    return dt

def track_criteria(dt, underlying):  #risk monitor parameters
    
    dt['contracts']=dt['con_1']+dt['con_2']-dt['con_ex1']-dt['con_ex2']
    dt['entry_price']=(dt['con_1']*dt['con_p1']+dt['con_2']*dt['con_p2'])/(dt['con_1']+dt['con_2'])
    dt['exit_price']=(dt['con_ex1']*dt['con_ex_p1']+dt['con_ex2']*dt['con_ex_p2'])/(dt['con_ex1']+dt['con_ex2'])
    con_m2m=pd.notnull(dt.exit_target)
    dt.loc[con_m2m,'erisk']= dt['contracts']*dt['exit_target']*100
    dt.loc[~con_m2m, 'erisk']=dt['contracts']*dt['entry_price']*100  #update exited trade before removal
    dt['epnl_pct']=0.01*dt['epnl']/((dt['con_1']+dt['con_2'])*dt['entry_price'])
    dt['epnl']=dt['contracts']*(dt['exit_target']-dt['entry_price'])*100-dt['comm']  #unrealized pnl
    dt['epnl_pct']=(dt['exit_target']/dt['entry_price']-1)
    dt['days_to_expire']=pd.to_datetime(dt['expire_date'])-pd.to_datetime(dt['date_last']) 
    dt['days_all_life']= pd.to_datetime(dt['expire_date'])-pd.to_datetime(dt['entry_date'])
# evaluate p_fwd_10 early sign    
    dt['days_from_entry']=dt['days_all_life'] - dt['days_to_expire']
    
# early indicator to stay or exit trade  !!
    dt.loc[dt['days_from_entry']=='5 days','p_5_fwd']=dt['close_last']
    dt.loc[dt['days_from_entry']=='10 days','p_10_fwd']=dt['close_last']
#        dt['p_fwd_10']=dt['close_last']
#    dt=win_loss(dt, 10)  # keep??
#    return dt
#    exit
    if underlying=='sp500': 
        con_datetime=(type(dt['ex_div'])==datetime.datetime)
        dt.loc[~con_datetime, 'ex_div']=datetime.datetime(2016,1,1) #day of month out of range error
    else: 
        dt['ex_div']=datetime.datetime(2016,1,1)
    dt['days_to_exdiv']=pd.to_datetime(dt['ex_div']) - pd.to_datetime(dt['date_last'])
    dt['exist_pct']=1-dt['days_to_expire']/dt['days_all_life']
    dt['exist_pct']=dt['exist_pct'].round(2)
    dt['days_left']=dt['days_to_expire']
    dt['days_to_exdiv']=dt['days_to_exdiv']
    #dt calculated fields
    dt['weigh']=dt['erisk']/(dt['erisk'].sum())
    #dt['pct_estrike_1']="{:.1%}".format(dt['close_last'])/dt['estike_1']-1)
    dt['pct_estrike_1']=(dt['close_last']/dt['estike_1']-1).round(2)
    dt['momt_n_chg']=dt['momt_n_last']-dt['momt_n']
    dt['trend_n_chg']=dt['trend_n_last']-dt['trend_n']
    dt['move_sig']=(dt['close_last']-dt['close_qdate'])/dt['p_22_sig'] 
    dt['rsi_chg']=dt['rsi']-dt['rsi_prev']
#alert_direction
    dt.loc[((dt['play']=='L')|(dt['play']=='LL')), 'alert_direction']=dt['rsi']<dt['rsi_prev'] #less momentum
    dt.loc[((dt['play']=='S')|(dt['play']=='SS')), 'alert_direction']=dt['rsi']>dt['rsi_prev']
    dt.loc[((dt['play']=='CAL')|(dt['play']=='ZZ')|(dt['play']=='Z0')|(dt['play']=='Z4')), 'alert_direction']=\
            np.abs(dt['pct_estrike_1'])>np.abs(dt['pct_estrike_prev'])
    dt.loc[dt['play']=='E', 'alert_direction']=np.abs(dt['pct_estrike_1'])<np.abs(dt['pct_estrike_prev'])
    dt.loc[dt['alert_stop']==True, 'counter']=dt['counter'].astype(float)+1  #counter

    con_too_late=(dt.exist_pct>=0.5) & (dt['exit_target']/dt['entry_price']<=1.1)
    con_stop_loss=dt.epnl_pct<= -0.35 # loss of 30%
    con_stop=con_too_late| con_stop_loss# | con_lastweek
    con_exit=(dt['play'].str.strip().str[0]=='Z')& (dt.pct_estrike_1 <=0.02) &(dt.exist_pct>=0.5)
    con_sig=(dt.close_last > dt.be_up) | (dt.close_last< dt.be_down) | (np.abs(dt.close_last - dt.close_qdate) >=dt.p_22_sig)
    con_div=(dt.days_to_exdiv<='5 days') & (dt.days_to_exdiv>='0 days')
    con_risk_allo=(dt.erisk/capital >0.15) | (dt.weigh > 0.35)
    con_prexit=pd.isnull(dt.alert_prexit) & dt.exist_pct>0.5  # pre-exit order
    
    dt['alert_exit']=con_exit
    dt['alert_sig']=con_sig
    dt['alert_stop']=con_stop
    dt['alert_div']=con_div
    


#display cleaning    
    dt.alert_direction[dt.alert_direction==False]="" 
    dt.alert_stop[dt.alert_stop==False]=""
    dt.alert_exit[dt.alert_exit==False]=""
    dt.alert_div[dt.alert_div==False]=""  
    dt.alert_sig[dt.alert_sig==False]=""  
#    dt.win_5[dt.win_5==False]="" 
#    dt.win_10[dt.win_10==False]="" 

    dt=dt.drop(['level_0'],1)
    if underlying=='sp500':
        to_sql_replace(dt, 'tbl_trade_track')
        to_sql_replace(dt, 'tbl_trade')
    else:
        to_sql_replace(dt, 'tbl_trade_etf')
    #pd.set_option('display.max_columns', None)

def track_show(dt):
    risk_allo=dt[con_risk_allo][['ticker', 'weigh', 'erisk']]
    risk_sig=dt[con_sig][['ticker','p_22_sig', 'move_sig','close_last', 'close_qdate','estike_1', 'be_up', 'be_down']]    
    risk_stop=dt[con_stop][['ticker', 'epnl_pct', 'exist_pct']]
    risk_exit=dt[con_exit][['ticker','epnl_pct', 'pct_estrike_1','exist_pct']]
    risk_div=dt[con_div][['ticker','ex_div']]
    risk_prexit=dt[con_prexit][['ticker','epnl_pct', 'exist_pct', 'days_to_expire']]
    
    print (colored('allocation risk', 'magenta', 'on_cyan'))    
    print(colored(risk_allo,'magenta'))
    print(" ------ ")
    print (colored('std deviation risk', 'red', 'on_cyan'))    
    print(colored(risk_sig,'blue'))
    print(" -------")
    print (colored('stop loss: HEDGED SURE PROFIT BY EXPIRE???', 'red','on_cyan'))    
    print(colored(risk_stop,'red'))
    print(" ------")
    print (colored('profit exit', 'red', 'on_green'))    
    print(colored(risk_exit,'green'))
    print("  -----    ")
    print (colored('need prexit order', 'red', 'on_cyan'))    
    print(colored(risk_prexit,'blue'))
    print("  -----    ")
    print (colored('event risk', 'magenta', 'on_cyan'))    
    print(colored(risk_div,'magenta'))
    print("")
    #print (colored('>>>>>>>>>     ALERT!!!    <<<<<<<<', 'red'))
    #print (df_alert[['ticker', 'alert_stop', 'alert_exit','erisk']])
    dtt=dt[['ticker','erisk', 'play', 'weigh', 'delta','beta', 'epnl', 'epnl_pct', 'alert_stop','alert_sig', 'alert_exit', 'alert_div', 'days_left',\
    #'alert_direction',\
    'pct_estrike_1','rsi_chg', 'p_pct', 'v_pct', 'si', 'ex_div', 'ta',  \
    'momt_n_chg', 'trend_n_chg', 'win_5', 'win_10', 'sec', 'rrtn_22_ts', 'rrtn_22_sm']]
    dtt.sort_values(['erisk', 'play'], ascending=[False, True], inplace=True)
    try:
        SPY=web.get_quote_yahoo('SPY')['last']
    except:
        print("web.get_quote_yahoo error@line 289")
        SPY=1000
    print (colored('>>>> SPY equivalent number of share in movement <<<<<<<<', 'red'))
    print("SPY equivalent share in movement:", (dt['beta']*dt['delta']*dt['close_last']).sum()/SPY)
    

