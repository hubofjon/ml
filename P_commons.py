# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 11:50:54 2017
purpose: common utility
Functions:
    1. to_sql_append
    2. to_sql_replace
    3. rad_sql
    4. email
    5. tbl_change
    6. drop_table
@author: jon
"""
import sqlite3 as db
import pandas as pd
import pandas.io.sql as pd_sql
import datetime
from shutil import copyfile
from timeit import default_timer as timer
from dateutil import parser

#db_str='c:\\pycode\db_op.db'
db_str="C:\\Users\qli1\BNS_git\git.db"

def bdate():
    today=datetime.datetime.today()
    if (today-pd.tseries.offsets.BDay(0))>today:
        bdate=(today-pd.tseries.offsets.BDay(1)).date()
    else:
        bdate=(today-pd.tseries.offsets.BDay(0)).date()
    return bdate

def to_sql_append(df, tbl_name):
    conn=db.connect(db_str)
    pd_sql.to_sql(df, tbl_name, conn, if_exists='append')
    conn.close()

def to_sql_replace(df, tbl_name):
    conn=db.connect(db_str)
    pd_sql.to_sql(df, tbl_name, conn, if_exists='replace')  
    conn.close()
    
#tbl_name is defined in query    
def read_sql(query, q_date=''):
    conn=db.connect(db_str)
    df=pd_sql.read_sql(query, conn)
    conn.close()
    return df

def to_sql_delete(query):
    conn=db.connect(db_str)
#    pd_sql.to_sql(df, tbl_name, conn, if_exists='replace')  
    cur=conn.cursor()
    cur.execute(query)
    conn.commit()
    print("to_sql_delete commited")
    cur.close()
    conn.close()

def backup_db():
    db_source=r'c:\pycode\db_op.db'
    db_dest=r'c:\pycode\db_op_bkup.db'
    copyfile(db_source, db_dest)    
    print("--- db_op is backup ----")

def type_convert(df, cols, type='float'):
    for x in cols:
        if type=='float':
            df[x]=df[x].astype(type)
        elif type=='datetime':
             df[x]=pd.to_datetime(df[x])
    return df    

def fmt_map(df, cols, fmt='{:.2f}'):
    df[cols]=df[cols].applymap(fmt.format)
#    format_mapping={'Currency': '${:,.2f}', 'Int': '{:,.0f}', 'Rate': '{:.2f}%'
#                    for key, value in format_mapping.items():
#   ....:     df[key] = df[key].apply(value.format)
    return df
#def sec_merge(df):
#    if 'sec' in df.columns:
#        print("sec_merge: sec already in df")
#        return df
#    ds=read_sql("SELECT * FROM tbl_sp_list")
#    df=pd.merge(df, ds[['ticker','sec']], how='left', on='ticker')
#    return df
    
def get_profile(df):
    ds=read_sql("SELECT * FROM tbl_pro")
    df=pd.merge(df, ds[['ticker','sec','beta','mkt']], how='left', on='ticker')
    return df

def tbl_nodupe_append(q_date, df, tbl=''):
    date_exist=read_sql("SELECT DISTINCT date FROM %s"%tbl,q_date)
    date_exist['date']=pd.to_datetime(date_exist['date']).dt.date
              
    if q_date in date_exist.date.unique():
        print("%s exists in %s already"%(q_date,tbl))
    else:
        to_sql_append(df, tbl)
        print("%s is appended for %s"%(tbl, q_date))
    
    
def tbl_dedupe(tbl,q_date):
    qry_count="SELECT date, COUNT(*) from '%s' \
    WHERE rowid not in (select min(rowid) \
    from '%s' group by date,ticker) "%(tbl,tbl)
    df_dupecnt=read_sql(qry_count, q_date)
    if ~df_dupecnt.empty:
        qry_del="DELETE from '%s' \
        WHERE rowid not in (select min(rowid) \
        from '%s' group by date,ticker) "%(tbl,tbl)    
        conn=db.connect('c:\\pycode\db_op.db')
        cur=conn.cursor()
        cur.execute(qry_del)
        conn.commit()
        cur.close()
    print("dates removed:  \n", df_dupecnt)

    #__________________UTILITY ZONE _______________________
def tbl_change():
    #add one column eaccdcdh time
    conn=db.connect(db_str)
    cur=conn.cursor()
#add new column

#    cur.execute("ALTER TABLE tbl_c add column i_play REAL")
#    cur.execute("ALTER TABLE tbl_c add column i_lsnv REAL")
#    cur.execute("ALTER TABLE tbl_trade_candy add column beta REAL")  
#    cur.execute("ALTER TABLE tbl_c add column sec REAL")
#    cur.execute("ALTER TABLE tbl_c_hist add column sec REAL")
#    cur.execute("ALTER TABLE tbl_mc_raw add column sec REAL")
#    cur.execute("ALTER TABLE tbl_mc_candy add column sec REAL")
#    cur.execute("ALTER TABLE tbl_bc_raw add column sec REAL")
#    cur.execute("ALTER TABLE tbl_bc_candy add column sec REAL")
#    cur.execute("ALTER TABLE tbl_bmc_candy add column sec REAL")
#    cur.execute("ALTER TABLE tbl_candies add column sec REAL")

#create index
    #cur.execute("CREATE UNIQUE INDEX date_index on tbl_price(date)")
#delete record_
    #cur.execute("DELETE FROM tbl_play_etf")    
    #drop one column
    #cur.execute("ALTER TABLE tbl_price_20161104 drop 'ALL'")
#COPY TABLE
#    cur.execute("CREATE TABLE tbl_c_190121 AS SELECT * FROM tbl_c")
#    cur.execute("CREATE TABLE tbl_stat_macro AS SELECT * FROM tbl_stat_etf")
    #change table columns type
#    cur.execute("CREATE TABLE tbl_TD ('index' INTEGER, alert_exit REAL,  alert_stop REAL,  be_down REAL,\
#    be_up REAL,  \
#    close_last REAL,\
#    close_qdate REAL,  contracts INTEGER,  date TIMESTAMP,  date_last TIMESTAMP,  entry_date TIMESTAMP,\
#    entry_note TEXT,  entry_price REAL,  epnl REAL,  epnl_pct REAL,  erisk REAL,  estike_1 REAL,  estrike_2 REAL,\
#    exist_pct REAL,  exit_date REAL,  exit_note REAL,  exit_price REAL,  exit_target REAL,  \
#    expire_date TIMESTAMP,\
#    hv_22 REAL, hv_252 REAL,  hv_m2y REAL,  mean_1022 REAL,  mean_2266 REAL, mean_66252 REAL,  \
#    p_22_sig REAL, p_44_sig REAL,  p_66_sig REAL,  play TEXT,  ticker TEXT,, earning_date TIMESTAMP, con_1 INTEGER, con_2 INTEGER, con_ex1 INTEGER, con_ex2 INTEGER, con_p1 REAL, con_p2 REAL, con_ex_p1 REAL, con_ex_p2 REAL, comm REAL)")
    #cur.execute("CREATE TABLE tbl_t1 (d1 INTEGER, d2 INTEGER, n1 REAL, n2 REAL, n3 INTEGER, d3 TIMESTAMP, d4 TIMESTAMP)")
#drop duplicate rows
#    cur.execute("DELETE FROM tbl_price_etf WHERE rowid not in (SELECT max(rowid) from tbl_price_etf group by date)")
#insert record
#    cur.execute("INSERT INTO tbl_stat_etf_hist(ticker,date) VALUES ('SPY','2018-04-02')")


    conn.commit()
    print ("tbl changed")
    cur.close()

def drop_table(tbl_name):
    conn=db.connect('c:\\pycode\db_op.db')
    cur=conn.cursor()
    cur.execute("DROP TABLE if exists '%s'"%tbl_name)
    conn.commit()
    cur.close()
    print("table: '%s' is dropped"%tbl_name)
#__________________COMMAND ZONE_________________________________
#drop_table('tb_trade')

# test_stat_prep: get df_stat for past date range, call stat_run, play_run, play_fwd_price, win_loss

# ----- test stratey thought as of Jan, 1, 2017
# 1. test_stat_prep() to get the df_stat (playbook_test is blank)
# 2. tbl_stat_test, 
# 3. plot_feature to visualize data first

def block_print():
    sys.stdout=open(os.devnull, 'w')

def enable_print():
   
    sys.stdout = old_stdout
    
def email():
    import smtplib
    from email.MIMEMultipart import MIMEMultipart
    from email.MIMEText import MIMEText
    from email.MIMEBase import MIMEBase
    from email import encoders
     
    fromaddr = ""
    toaddr = "EMAIL ADDRESS YOU SEND TO"
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "SUBJECT OF THE EMAIL"
    body = "TEXT YOU WANT TO SEND"
     
    msg.attach(MIMEText(body, 'plain'))
     
    filename = "NAME OF THE FILE WITH ITS EXTENSION"
    attachment = open("PATH OF THE FILE", "rb")
     
    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
     
    msg.attach(part)
     
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, "YOUR PASSWORD")
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()
    
    #__________RESOURCE ZONE  ____________________________________
    
#https://pypi.python.org/pypi/googlefinance
#https://pypi.python.org/pypi/yahoo-finance/1.1.4
#http://www.sqlite.org/lang_datefunc.html
#df.loc[<row selection>, <column selection>]
#w.loc[w.female != 'female', 'female'] = 0
#  http://machinelearningmastery.com/feature-selection-machine-learning-python/  
#dy=pd.rolling_corr(dx['SPY'], dx, window=66, pairwise=True)
#dy.mean()
#dy.mean(axis=1)
#Python-Financial-Tools/capm.py  *capm model
#https://github.com/jealous/stockstats    
#http://ahmedbesbes.com/how-to-mine-newsfeed-data-and-extract-interactive-insights-in-python.html   
#    https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/