# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 13:02:52 2019

@author: qli1
"""

import pandas as pd
import numpy as np
import datetime as datetime
import pandas_datareader.data as web
from pandas_datareader import data, wb
import sys, os
import sqlite3 as db
import pandas.io.sql as pd_sql
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec   
import warnings
pd.set_option('display.expand_frame_repr', False)
#from P_commons import read_sql
pd.options.display.float_format = '{:,.2f}'.format
warnings.filterwarnings("ignore")

#Spike profile, n=hv windows, s=sample days
#plot_fulls/ full, plot_pv (basic plot)
def read_sql(query):
    conn=db.connect('C:\\Users\qli1\BNS_git\git.db')
    df=pd_sql.read_sql(query, conn)
    conn.close()
    return df

def get_data(q_date):
    dbc=read_sql("SELECT * FROM tbl_bc_raw" )
    dmc=read_sql("SELECT * FROM tbl_mc_raw" )
    dbc['date']=pd.to_datetime(dbc['date'])
    dmc['date']=pd.to_datetime(dbc['date'])
    dates=dbc.date
#    dbc=dbc[dbc.date==q_date]
#    dmc=dmc[dmc.date==q_date]
   
    return dbc, dmc

dte_mins=[1, 5,20,60]
def senti(q_date, df_grp):
    #q_date, df (without date columns), return ds@q_date
#    dte_mins=[1, 5,20,60]

    p_min=10
    pct_c_min=70
    v_opt_min=1000
    
    #Filter for conviction    
    ds_data=[]
    for x in  np.arange(len(dte_mins)):
        df=df_grp.sort_values('ticker')  #df muted at each iteration!
        if x<= (len(dte_mins)-2):
            con_dte= (dte_mins[x]<= df['dte']) & (df['dte'] < dte_mins[x+1])
        else:
            con_dte= (dte_mins[x]<= df['dte'])
        con_p=df_grp['price']>p_min  
        df=df[con_dte & con_p]
        df['prem']=df['last']* df['vol']
        con_c=(df['type'].str.upper()=='CALL')
        con_p=(df['type'].str.upper()=='PUT') 
        con_bc=(df['type'].str.upper()=='CALL') & (df['bs']=='b')
        con_sc=(df['type'].str.upper()=='CALL') & (df['bs']=='s')    
        con_bp=(df['type'].str.upper()=='PUT') & (df['bs']=='b')
        con_sp=(df['type'].str.upper()=='PUT') & (df['bs']=='s')   
        prem_c='{:.1f}'.format(df[con_c].prem.sum())
        prem_p='{:.1f}'.format(df[con_p].prem.sum())
        prem_bc='{:.1f}'.format(df[con_bc].prem.sum())
        prem_sc='{:.1f}'.format(df[con_sc].prem.sum())    
        prem_bp='{:.1f}'.format(df[con_bp].prem.sum())
        prem_sp='{:.1f}'.format(df[con_sp].prem.sum())      
        prem_bcsp=(float(prem_bc)+ float(prem_sp))
        prem_scbp=(float(prem_sc)+ float(prem_bp))  
        
        if float(prem_p)!=0:
            pr_CP= '{:.1f}'.format(float(prem_c)/float(prem_p))
        else:
            pr_CP=np.nan
            
        if (float(prem_bp) + float(prem_sp))!=0:
            pr_cp= '{:.1f}'.format(float((float(prem_bc)+ float(prem_sc))/(float(prem_bp) + float(prem_sp))))
        else:
            pr_cp=np.nan
            
        if float(prem_scbp) !=0:
            pr_bubr='{:.1f}'.format(float(float(prem_bcsp)/float(prem_scbp)))
        else:
            pr_bubr=np.nan
            
        if float(prem_bp) !=0:
            pr_bcbp='{:.1f}'.format(float(float(prem_bc)/float(prem_bp)))
        else:
            pr_bcbp=np.nan
            
        if float(prem_sp) !=0:
            pr_scsp='{:.1f}'.format(float(float(prem_sc)/float(prem_sp)))
        else:
            pr_scsp=np.nan
        
        if float(prem_c) !=0:
            pr_sc='{:.1f}'.format(float(float(prem_sc)/float(prem_c)))
        else:
            pr_sc=np.nan
            
        if float(prem_p) !=0:
            pr_sp='{:.1f}'.format(float(float(prem_sp)/float(prem_p)))
        else:
            pr_sp=np.nan
    #    pr_bcbp=(prem_bc/prem_bp).round(2)   
        date=q_date
        ds_tmp=[date, prem_bc, prem_sc, prem_bp, prem_sp, pr_cp, pr_bubr, \
                pr_bcbp, pr_scsp, dte_mins[x], prem_c, prem_p, pr_CP, pr_sc, pr_sp]
        ds_data.append(ds_tmp)
#    return bc_data
    ds_data=np.asarray(ds_data)
    ds_data=ds_data.transpose()
    ds_columns=['date','prem_bc', 'prem_sc', 'prem_bp', 'prem_sp', 'pr_cp', 'pr_bubr', \
        'pr_bcbp', 'pr_scsp','dte_min', 'prem_c','prem_p','pr_CP','pr_sc','pr_sp']
#    ds_bc=pd.DataFrame(bc_data, columns=bc_columns, index=np.arange(1))

    ds=pd.DataFrame(dict(zip(ds_columns, ds_data)), index=np.arange(len(dte_mins)))
    cols= ['prem_bc','prem_sc', 'prem_bp', 'prem_sp', 'pr_cp', 'pr_bubr',\
         'pr_bcbp', 'pr_scsp', 'dte_min','prem_c','prem_p','pr_CP', 'pr_sc','pr_sp']   
    ds=convert(ds, cols, 'float')
    return ds
    #mc
#    con_p_mc=dmc['p'].astype(float)> p_min
#    con_v_opt=dmc['v_opt'].astype(float)> v_opt_min
#    dmc=dmc[con_p_mc & con_v_opt]

def sentis():
    dbc=read_sql("SELECT * FROM tbl_bc_raw" )
    dmc=read_sql("SELECT * FROM tbl_mc_raw" )
    dates=dbc.date.unique()
    dbc_grp=dbc.groupby('date')
    dmc_grp=dmc.groupby('date')
    ds=pd.DataFrame()
    for d,g in dbc_grp:
        ds_tmp=senti(d, g)
        ds=pd.concat([ds, ds_tmp], axis=0)
    
    ds['date']=pd.to_datetime(ds['date'])
    dp=read_sql("SELECT * FROM tbl_pv_etf WHERE ticker='SPY'")
    dp['date']=pd.to_datetime(dp['date'])
    dp['rtn']=dp['close'].pct_change()
    dp['rtn_1']=dp['close'].pct_change().shift(1)
    dp['rtn_5']=dp['close'].pct_change().shift(5)
    dp['rtn_20']=dp['close'].pct_change().shift(20)
#    dp['rtn_log']=np.log(1+dp['close'].pct_change())
    dp=dp[dp.date.isin(ds.date)]
    dsp=pd.merge(ds, dp[['date','rtn','rtn_1','rtn_5', 'rtn_20','close']], on='date')
#    dsp=pd.merge(ds, dp[['date','rtn','rtn_log']].shift(1), on='date')
#SHIFT rtn one day forward to test predicting correlation 
    for d in dte_mins:
        df=dsp[dsp['dte_min']==d]
        corr=df[['pr_cp', 'pr_bubr','pr_bcbp', 'pr_scsp', 'pr_CP', 'pr_sc','pr_sp',\
            'rtn', 'rtn_1','rtn_5','rtn_20']].corr (method='pearson')
        print('\n corr_%s: \n'%d, corr )
    return dsp
#    for d in dates.values:
#        senti()
def senti_intel(df, q_date): #df =dsp , output of sentis()
    col_pr=['pr_CP', 'pr_bcbp', 'pr_bubr', 'pr_cp', 'pr_sc','pr_scsp', 'pr_sp']
    col_prem=['prem_bc', 'prem_bp', 'prem_c', 'prem_p', 'prem_sc', 'prem_sp']
    #stat @q_date
    dx=df[df.date==q_date]
    dx.fillna(0, inplace=True)
    dx['bc_pct']=dx['prem_bc']/(dx['prem_bc'].sum())
    dx['sc_pct']=dx['prem_sc']/(dx['prem_sc'].sum())
    dx['bp_pct']=dx['prem_bp']/(dx['prem_bp'].sum())
    dx['sp_pct']=dx['prem_sp']/(dx['prem_sp'].sum())
    show=['dte_min', 'pr_CP', 'pr_cp', 'pr_bcbp', 'pr_bubr',  'pr_sc',\
       'pr_scsp', 'pr_sp', 'prem_bc', 'prem_bp', 'prem_c', 'prem_p', 'prem_sc',\
       'prem_sp', 'bc_pct', 'sc_pct', 'bp_pct', 'sp_pct']
    s=dx[col_prem].sum()
    s['pr_CP']=s['prem_c']/s['prem_p']
    s['pr_cp']=(s['prem_bc'] + s['prem_sc'])/(s['prem_bp'] + s['prem_sp'])
    s['pr_bubr']=(s['prem_bc'] + s['prem_sp'])/(s['prem_sc'] + s['prem_bp'])
    s['pr_bcbp']=s['prem_bc']/s['prem_bp']
    s['pr_scsp']=s['prem_sc']/s['prem_sp']
    s['pr_sc']=s['prem_sc']/s['prem_c']
    s['pr_sp']=s['prem_sp']/s['prem_p']
    s['date']=q_date
#dy: @q_date, summary of prem, pr, and sum
#dx: summary of complete data set in tbl_bc_raw
    col_prem=['prem_bc', 'prem_bp', 'prem_c', 'prem_p', 'prem_sc','prem_sp']
    dy=dx.append(s, ignore_index=True)
#get describe stat by dte_min
    ds=pd.DataFrame()
    dey=pd.DataFrame()
    for x in dte_mins:
        df=dx[dx.dte_min==x]
        ds_tmp=df[col_pr].describe()
        ds_tmp=des_tmp.iloc[[4,6],:]
        ds_tmp['dte_min']=x
        ds=pd.concat([ds, ds_tmp], axis=0)

        dey_pr=dy[dte_min==x][col_pr]
        dey_prem=dy[dte_min==x][col_prem]
        con_25= dey_pr< des_tmp.iloc[0]
        con_75= dey_pr> des_tmp.iloc[1]      
        con= (con_25 | con_75)
        dey_tmp=pd.concat([dey_pr[con], dey_prem], axis=1)
        dey=pd.concat([dey, dey_tmp], axis=0)
#    print(dx[show])
    return dy, ds
    
    
    

def convert(df, cols, type='float'):
    for x in cols:
        df[x]=df[x].astype(type)
    return df