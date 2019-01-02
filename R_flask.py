# -*- coding: utf-8 -*-
from flask import Flask, request, render_template, url_for, redirect, session, flash, g
import os
import sqlite3
import pandas.io.sql as pd_sql

host = os.getenv("IP", "127.0.0.1")
port = int(os.getenv("PORT", "8080"))

# configuration details
#DATABASE = "C:\\Users\qli1\BNS_wspace\flask\f_trd\trd.db"
#DATABASE = "C:\\pycode\db_op_20170907"
SECRET_KEY = "my_secret"

app = Flask(__name__)
app.config.from_object(__name__) # pulls in app configuration by looking for uppercase variables

def connect_db():
#    return sqlite3.connect(app.config["C:\Users\qli1\BNS_wspace\flask\f_books\books.db"])
    connect=sqlite3.connect(r"C:\pycode\db_op.db")
    return connect
    
@app.route("/main")
def main():
    # create connection to db
    g.db = connect_db()
    trades = []
    df=pd_sql.read_sql("SELECT * FROM tbl_c", g.db)
    for index, row in df.iterrows():  #"trd_id":df.index.values[index],
        trades.append({ "ticker":df.loc[index,'ticker'],  "mp":df.loc[index,'mp'], \
            "exit_dt":df.loc[index,'exit_dt'],"tgt_dt":df.loc[index,'tgt_dt'],"tgt_p":df.loc[index,'tgt_p'],\
            "ic1":df.loc[index,'ic1'],"ip1":df.loc[index,'ip1'],"ic2":df.loc[index,'ic2'],"ip2":df.loc[index,'ip2'],\
            "comm":df.loc[index,'comm'],\
            "oc1":df.loc[index,'oc1'],"op1":df.loc[index,'op1'],"oc2":df.loc[index,'oc2'],"op2":df.loc[index,'op2'] })
    # pass books data to our template for use
    return render_template("main.html", trades=trades)

# edit route
@app.route("/edit", methods=["GET", "POST"])
def edit():
    msg = ""
    # get query string arguments
    ticker = request.args.get("ticker")
    tgt_dt = request.args.get("tgt_dt")
    exit_dt = request.args.get("exit_dt")
    mp = request.args.get("mp")
    tgt_p = request.args.get("tgt_p")
    ic1 = request.args.get("ic1")
    ip1 = request.args.get("ip1")
    ic2 = request.args.get("ic2")
    ip2 = request.args.get("ip2")
    comm = request.args.get("comm")
    oc1 = request.args.get("oc1")
    op1 = request.args.get("op1")
    oc2 = request.args.get("oc2")
    op2 = request.args.get("op2")
    
    trd = {
#        "trd_id": trd_id,
        "ticker":ticker,
        "tgt_dt":tgt_dt,
        "exit_dt":exit_dt,
        "mp":mp,
        "tgt_p":tgt_p,
        "ic1":ic1,
        "ip1":ip1,
        "ic2":ic2,
        "ip2":ip2,
        "comm":comm,
        "oc1":oc1,
        "op1":op1,
        "oc2":oc2,
        "op2":op2        
    }
    
    if request.method == "POST":
        # get the data from form
 #       trd_id = request.form["trd_id"]
        ticker = request.form["ticker"]
        tgt_dt = request.form["tgt_dt"]
        exit_dt = request.form["exit_dt"]
        mp = request.form["mp"]
        tgt_p= request.form["tgt_p"]
        ic1 = request.form["ic1"]
        ip1 = request.form["ip1"]
        ic2 = request.form["ic2"]
        ip2 = request.form["ip2"]
        comm= request.form["comm"]
        oc1 = request.form["oc1"]
        op1 = request.form["op1"]
        oc2 = request.form["oc2"]
        op2 = request.form["op2"]
        # connect db and update record
        g.conn = connect_db()
        cursor = g.conn.cursor()
        cursor.execute("UPDATE tbl_c SET ticker=?, tgt_dt=?, exit_dt=?, mp=?, tgt_p=?\
                       ,ic1=?, ip1=?, ic2=?, ip2=?, comm=?,oc1=?, op1=?, oc2=?, op2=? \
                       WHERE ticker=?", (ticker, tgt_dt, exit_dt, mp, tgt_p, ic1, ip1, ic2, ip2, \
                                         comm, oc1, op1, oc2, op2, ticker))
        g.conn.commit()
        g.conn.close()
        trd = {
   #         "trd_id": trd_id,
            "ticker":ticker,
            "tgt_dt":tgt_dt,
            "exit_dt":exit_dt,
            "mp":mp,
            "tgt_p":tgt_p,
            "ic1":ic1,
            "ip1":ip1,
            "ic2":ic2,
            "ip2":ip2,
            "comm":comm,
            "oc1":oc1,
            "op1":op1,
            "oc2":oc2,
            "op2":op2             
        }
        
        msg = "Record successfully updated!"
    
    return render_template('edit.html', t=trd, message=msg)
    
if __name__ == "__main__":
    app.run(host=host, port=port, debug=True)
        