import os
import sqlite3 as db_test
import pandas as pd
import pandas.io.sql as pd_sql
#from P_commons import get_conn

#data_url = 'https://people.sc.fsu.edu/~jburkardt/data/csv/addresses.csv'
#headers = ['first_name','last_name','address','city','state','zip']
#data_table = pd.read_csv(data_url, header=None, names=headers, converters={'zip': str})

# Clear example.db if it exists
#if os.path.exists('example.db'):
#    os.remove('example.db')
db_str=r"C:\Users\qli1\BNS_git\git.db"
#example_str='c:\\pycode\db_op.db'
## Create a database
con = db_test.connect(db_str, check_same_thread=False)
#conn=get_conn()

# Add the data to our database
#data_table.to_sql('data_table', conn, dtype={
#    'first_name':'VARCHAR(256)',
#    'last_name':'VARCHAR(256)',
#    'address':'VARCHAR(256)',
#    'city':'VARCHAR(256)',
#	'state':'VARCHAR(2)',
#	'zip':'VARCHAR(5)',
#})

con.row_factory = db_test.Row

# Make a convenience function for running SQL queries
def sql_query(query):
    cur = con.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    return rows

def sql_edit_insert(query,var):
    cur = con.cursor()
    cur.execute(query,var)
    con.commit()

def sql_delete(query,var):
    cur = con.cursor()
    cur.execute(query,var)
    con.commit()
    
def sql_query_var(query,var):
    cur = con.cursor()
    cur.execute(query,var)
    rows = cur.fetchall()
    return rows

def sql_query_one(query,var):
    cur = con.cursor()
    cur.execute(query,var)
    rows = cur.fetchone()
    return rows

def sql_read(query):
    df=pd_sql.read_sql(query, con)
    return df

def sql_append(df, tbl_name):
    pd_sql.to_sql(df, tbl_name, con, if_exists='append', index=False)

def sql_replace(df, tbl_name):
    pd_sql.to_sql(df, tbl_name, con, if_exists='replace', index=False)  
