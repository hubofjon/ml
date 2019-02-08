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
warnings.filterwarnings("ignore")

#Spike profile, n=hv windows, s=sample days
#plot_fulls/ full, plot_pv (basic plot)
def read_sql(query):
    conn=db.connect('C:\\Users\qli1\BNS_git\git.db')
    df=pd_sql.read_sql(query, conn)
    conn.close()
    return df

def get_data():
    dbc=read_sql("SELECT * FROM tbl_bc_raw" )
    dmc=read_sql("SELECT * FROM tbl_mc_raw" )
    dbc['date']=pd.to_datetime(dbc['date'])
    dmc['date']=pd.to_datetime(dbc['date'])
    return dbc, dmc

def senti():
    dbc, dmc=get_data()
    #bc
    dte_mins=[5,20,60]
    p_min=10
    pct_c_min=70
    v_opt_min=1000
    con_p_bc=dbc['price']>p_min
    #Filter for conviction    
    bc_data=[]
    for d in dte_mins:
        df=dbc.sort_values('ticker')
        con_dte=df['dte']>= d
        df=df[con_dte & con_p_bc]
        df['prem']=df['last']* df['vol']
        con_bc=(df['type'].str.upper()=='CALL') & (df['bs']=='b')
        con_sc=(df['type'].str.upper()=='CALL') & (df['bs']=='s')    
        con_bp=(df['type'].str.upper()=='PUT') & (df['bs']=='b')
        con_sp=(df['type'].str.upper()=='PUT') & (df['bs']=='s')   
        prem_bc=df[con_bc].prem.sum()
        prem_sc=df[con_sc].prem.sum()    
        prem_bp=df[con_bp].prem.sum()
        prem_sp=df[con_sp].prem.sum()       
        prem_bcsp=(prem_bc+ prem_sp)
        prem_scbp=(prem_sc+ prem_bp)  
        pr_cp= (prem_bc+ prem_sc)/(prem_bp + prem_sp)
        pr_bubr=(prem_bcsp/prem_scbp)
        pr_bcbp=(prem_bc/prem_bp)
    #    pr_bcbp=(prem_bc/prem_bp).round(2)   
        bc_tmp=[prem_bc, prem_sc, prem_bp, prem_sp, pr_cp, pr_bubr, pr_bcbp, d]
        bc_data.append(bc_tmp)
#    return bc_data
    bc_columns=['prem_bc', 'prem_sc', 'prem_bp', 'prem_sp', 'pr_cp', 'pr_bubr', \
                'pr_bcbp', 'dte_min']
#    ds_bc=pd.DataFrame(bc_data, columns=bc_columns, index=np.arange(1))
    ds_bc=pd.DataFrame(dict(zip(bc_columns, bc_data)), index=np.arange(len(dte_mins)))
    show_bc=['dte_min','pr_bcbp', 'pr_bubr','pr_cp']
    print(ds_bc[show_bc])
    #mc
    con_p_mc=dmc['p'].astype(float)> p_min
    con_v_opt=dmc['v_opt'].astype(float)> v_opt_min
    dmc=dmc[con_p_mc & con_v_opt]

