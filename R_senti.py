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
    con_p=df_grp['price']>p_min
    #Filter for conviction    
    ds_data=[]
    for x in  np.arange(len(dte_mins)):
        df=df_grp.sort_values('ticker')  #df muted at each iteration!
        if x<= (len(dte_mins)-2):
            con_dte= (dte_mins[x]<= df['dte']) & (df['dte'] < dte_mins[x+1])
        else:
            con_dte= (dte_mins[x]<= df['dte'])
            
        df=df[con_dte & con_p]
        df['prem']=df['last']* df['vol']
        con_bc=(df['type'].str.upper()=='CALL') & (df['bs']=='b')
        con_sc=(df['type'].str.upper()=='CALL') & (df['bs']=='s')    
        con_bp=(df['type'].str.upper()=='PUT') & (df['bs']=='b')
        con_sp=(df['type'].str.upper()=='PUT') & (df['bs']=='s')   
        prem_bc='{:.1f}'.format(df[con_bc].prem.sum())
        prem_sc='{:.1f}'.format(df[con_sc].prem.sum())    
        prem_bp='{:.1f}'.format(df[con_bp].prem.sum())
        prem_sp='{:.1f}'.format(df[con_sp].prem.sum())      
        prem_bcsp=(float(prem_bc)+ float(prem_sp))
        prem_scbp=(float(prem_sc)+ float(prem_bp))  
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
    #    pr_bcbp=(prem_bc/prem_bp).round(2)   
        date=q_date
        ds_tmp=[date, prem_bc, prem_sc, prem_bp, prem_sp, pr_cp, pr_bubr, \
                pr_bcbp, pr_scsp, dte_mins[x]]
        ds_data.append(ds_tmp)
#    return bc_data
    ds_data=np.asarray(ds_data)
    ds_data=ds_data.transpose()
    ds_columns=['date','prem_bc', 'prem_sc', 'prem_bp', 'prem_sp', 'pr_cp', 'pr_bubr', \
                'pr_bcbp', 'pr_scsp','dte_min']
#    ds_bc=pd.DataFrame(bc_data, columns=bc_columns, index=np.arange(1))

    ds=pd.DataFrame(dict(zip(ds_columns, ds_data)), index=np.arange(len(dte_mins)))
    cols= ['prem_bc','prem_sc', 'prem_bp', 'prem_sp', 'pr_cp', 'pr_bubr',\
           'pr_bcbp', 'pr_scsp', 'dte_min']   
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
    dp['rtn_log']=np.log(1+dp['close'].pct_change())
    dp=dp[dp.date.isin(ds.date)]
    dsp=pd.merge(ds, dp[['date','rtn','rtn_log']], on='date')
#    dsp=pd.merge(ds, dp[['date','rtn','rtn_log']].shift(1), on='date')
#SHIFT rtn one day forward to test predicting correlation 
    
    for d in dte_mins:
        df=dsp[dsp['dte_min']==d]
        corr=df[['pr_cp', 'pr_bubr','pr_bcbp', 'pr_scsp', 'rtn', 'rtn_log']].corr\
        (method='pearson')
        print('\n corr_%s: \n'%d, corr )

    return ds
        
#    for d in dates.values:
#        senti()

def convert(df, cols, type='float'):
    for x in cols:
        df[x]=df[x].astype(type)
    return df