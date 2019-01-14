# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 10:46:42 2017
Purpose:  stat_option
Functions: get_option
@author: jon
"""
import pandas as pd
import numpy as np
import datetime as datetime
import time
from P_commons import bdate
from P_commons import to_sql_append, to_sql_replace, read_sql
pd.options.display.float_format = '{:.2f}'.format
from termcolor import colored
from timeit import default_timer as timer
from T_intel import get_earning_ec, get_DIV, get_si_sqz, get_rsi, get_RSI
from P_commons import to_sql_append, read_sql
from R_plot import plot_base
from datetime import timedelta
from R_stat import stat_VIEW

def get_option_simple(ticker=''):
#ref: http://yizeng.me/2014/04/08/get-text-from-hidden-elements-using-selenium-webdriver/
    from lxml import html
    from selenium.webdriver.chrome.options import Options
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.keys import Keys
    from selenium.common.exceptions import TimeoutException

    chrome_options=Options()
    chrome_options.add_argument("--disable-popup")
    chrome_options.add_extension(r"c:\pycode\Github\extension_1_0_7_overlay_remove.crx")
    chrome_options.add_extension(r"c:\pycode\Github\extension_1_13_8.crx")  #fairad
    #chrome_options.add_extension(r"G:\Trading\Trade_python\pycode\Github\extension_0_3_4.crx")
    chrome_options.add_argument('--always-authorize-plugins=true')
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--start-maximized")  #full screen
    x=r"C:\Users\jon\AppData\Local\Google\Chrome\User Data\Default"
    chrome_options.add_argument("user-data-dir=%s"%x)
    url_1="https://marketchameleon.com/Overview/"+ticker
    url_2=url_1+"/OptionSummary/"
    url_3=url_1+"/StockPosts/"
    url_4=url_1+"/SymbolReview/"
    gecko="c:\\pycode\Github\chromedriver.exe"
    driver=webdriver.Chrome(executable_path="c:\pycode\Github\chromedriver.exe", \
        chrome_options=chrome_options)
#LOG IN
    url_0="https://marketchameleon.com/Account/Login"
#    driver.get(url_0)
#    time.sleep(4)
#    try:
#        ol=driver.find_element_by_class_name("register-flyer-close")
#        ol=driver.find_element
#        ol.click()
#    except:
#        print("no find flyer close")
#        pass
#    username = driver.find_element_by_id("UserName")
#    password = driver.find_element_by_id("Password")
#    username.send_keys("freeofjon@gmail.com")
#    password.send_keys("kevin2008")
#    submit=driver.find_element_by_xpath('//input[@class="site-login-control btn"]')
#    time.sleep(2)
#    submit.send_keys('\n')

#go to url_1
    driver.get(url_1)
    time.sleep(1)
    try:
        ol=driver.find_element_by_class_name("register-flyer-close")
        ol=driver.find_element
        ol.click()
    except:
        pass
    iv_30_rank=driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[3]/div[6]/span[2]').get_attribute('textContent')
    iv_30s=driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[3]/div[5]/span[2]').get_attribute('textContent')
    hv_1y=driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[3]/div[4]/span[2]').get_attribute('textContent')
    hv_22=driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[3]/div[3]/span[2]').get_attribute('textContent')

    v_stk=driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[2]/div[2]/span[2]').get_attribute('textContent')
    v_stk_avg=driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[2]/div[3]/span[2]').get_attribute('textContent')
    v_opt=driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[2]/div[4]/span[2]').get_attribute('textContent')
    v_opt_avg=driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[2]/div[5]/span[2]').get_attribute('textContent')
    v_stk=v_stk.replace(",","")
    v_stk_avg=v_stk_avg.replace(",","")
    v_opt=v_opt.replace(",","")
    v_opt_avg=v_opt_avg.replace(",","")
    v_stk_pct='{:.1%}'.format(float(v_stk)/float( v_stk_avg))
    try:
        v_opt_pct='{:.1%}'.format(float(v_opt)/ float(v_opt_avg))
    except:  #v_op is zero
        pass
#    etf_o=driver.find_element_by_xpath('//*[@id="mkt_corr_tbl"]/tbody/tr[1]/td[1]/span/a').get_attribute('textContent')
#    corr_etf=driver.find_element_by_xpath('//*[@id="mkt_corr_tbl"]/tbody/tr[1]/td[4]').get_attribute('textContent')
#    p_chg_etf=driver.find_element_by_xpath('//*[@id="mkt_corr_tbl"]/tbody/tr[1]/td[7]/span').get_attribute('textContent')
#    p_chg=driver.find_element_by_xpath('//*[@id="sym_heading_top"]/div[1]/div[2]/div[2]/p[3]').get_attribute('textContent')
#    p=driver.find_element_by_xpath('//*[@id="overview_last_price"]').get_attribute('textContent')
#    earn_date= driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[4]/div[4]/span[2]').get_attribute('textContent')
    ex_div= driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[4]/div[2]/span[2]').get_attribute('textContent')
    yld= driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[4]/div[3]/span[2]').get_attribute('textContent')
#    pe= driver.find_element_by_xpath('//*[@id="symov_main_sidebar"]/div[1]/div[4]/div[5]/span[2]').get_attribute('textContent')


    iv_30=float(iv_30s.split(" ")[0])
    iv_30_chg=float(iv_30s.split(" ")[1])
    iv_30_chg='{:.1%}'.format(iv_30_chg/ iv_30)
    iv_30_rank=iv_30_rank.split(" ")[0]
    iv_hv=float(iv_30)/float(hv_22)
    hv_rank=float(hv_22)/float(hv_1y)

#go to url_2: option summary for call, put volume
    driver.get(url_2)
    time.sleep(1)
    try:
        ol=driver.find_element_by_class_name("register-flyer-close")
        ol=driver.find_element
        ol.click()
    except:
        pass

    v_c= driver.find_element_by_xpath('//*[@id="option_summary_MainTotal"]/tbody/tr/td[6]').get_attribute('textContent')
    v_p= driver.find_element_by_xpath('//*[@id="option_summary_MainTotal"]/tbody/tr/td[7]').get_attribute('textContent')
    oi_c= driver.find_element_by_xpath('//*[@id="option_summary_MainTotal"]/tbody/tr/td[8]').get_attribute('textContent')
    oi_p= driver.find_element_by_xpath('//*[@id="option_summary_MainTotal"]/tbody/tr/td[9]').get_attribute('textContent')

    v_c=float(v_c.replace(",", ""))
    v_p=float(v_p.replace(",", ""))
    oi_c=float(oi_c.replace(",", ""))
    oi_p=float(oi_p.replace(",", ""))
    pcr_v=v_p/ v_c
    pcr_oi=oi_p/ oi_c #avoid division by zero

    names=['iv_30', 'iv_30_rank', 'iv_hv', 'hv_1y', 'hv_22', 'hv_rank',   \
        'v_stk', 'v_stk_avg', 'v_opt', 'v_opt_avg', 'v_stk_pct','v_opt_pct', \
        'ex_div', 'yld', 'pcr_v', 'pcr_oi']
    values=[iv_30, iv_30_rank, iv_hv,  hv_1y, hv_22, hv_rank,\
            v_stk, v_stk_avg, v_opt, v_opt_avg, v_stk_pct, v_opt_pct, \
       ex_div, yld, pcr_v, pcr_oi]

    do=pd.DataFrame(data=[], columns=names)
    do.loc[0,names]=values
    do['ticker']=ticker
    driver.quit()
#    driver.close
    return do

def get_unop_mc():  # take max 5 pages from unop  url
#ref: http://yizeng.me/2014/04/08/get-text-from-hidden-elements-using-selenium-webdriver/
    from lxml import html
    from selenium.webdriver.chrome.options import Options
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.keys import Keys
    from selenium.common.exceptions import TimeoutException
    from bs4 import BeautifulSoup

    chrome_options=Options()
    chrome_options.add_argument("--disable-popup")
    chrome_options.add_extension(r"c:\pycode\Github\extension_1_0_7_overlay_remove.crx")
    chrome_options.add_extension(r"c:\pycode\Github\extension_1_13_8.crx")  #fairad
    #chrome_options.add_extension(r"G:\Trading\Trade_python\pycode\Github\extension_0_3_4.crx")
    chrome_options.add_argument('--always-authorize-plugins=true')
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--start-maximized")  #full screen
    x=r"C:\Users\jon\AppData\Local\Google\Chrome\User Data\Default"
    chrome_options.add_argument("user-data-dir=%s"%x)
    url="https://marketchameleon.com/Reports/UnusualOptionVolumeReport"
    gecko="c:\\pycode\Github\chromedriver.exe"
    driver=webdriver.Chrome(executable_path="c:\pycode\Github\chromedriver.exe", \
        chrome_options=chrome_options)
    driver.get(url)

    time.sleep(3)
    html=driver.page_source
    soup=BeautifulSoup(html, 'lxml')
 #close flyer
    try:
        #flyer=soup.find("div",{"class":"register-flyer-close"})
        flyer=driver.find_element_by_css_selector('#register_flyer > div.register-flyer-close')
        flyer.click()
        flyer2=driver.find_element_by_id('register_flyer')
        flyer2.click()
    except:
        pass  
    
    
#find report date
    try:
        mc_date=soup.findAll("div",{"class":"report_as_of_time"})[0].text[13:]
    except: 
        mc_date=datetime.date.today().strftime("%Y-%m-%d")
 
#find subtitle
    titles=[]
    subs=soup.find("tr",{"class":"sub_heading"})
    for th_tags in subs.find_all('th'):
        titles.append(th_tags.get_text())

#find table contents
    table=soup.find("table",{"id":"opt_unusual_volume"})
    pages=[]
    pages.append(titles)   # APPEND whole list of subtile as first element
    aray=[]
    lis=[]
    for row in table.find_all('tr'):
        for td_tags in row.find_all('td'):
            x= td_tags.get_text()
            lis.append(x)
        aray.append(lis)
        lis=[]
    pages.extend(aray)
    nxt=driver.find_element_by_css_selector("a.paginate_button.next")
#    nxt_dis=driver.find_element_by_css_selector("a.paginate_button.next.disabled")
    page_num=0
    nxt_dis=False
    while (nxt and page_num<10 and (not nxt_dis)):
#        if page_num<10:
            try:
                nxt.send_keys(Keys.END)
            except:
                pass
    #        driver.execute_script("window.scrollTo(0,  document.body.scrollHeight);")
            time.sleep(2)
            try:
                nxt.click()
            except:
                driver.close()
                return pages, mc_date

            time.sleep(3)
            html=driver.page_source
            soup=BeautifulSoup(html, 'lxml')
            table=soup.find("table",{"id":"opt_unusual_volume"})
            aray=[]
            lis=[]
            for row in table.find_all('tr'):
                for td_tags in row.find_all('td'):
                    x= td_tags.get_text()
                    lis.append(x)
                aray.append(lis)
                lis=[]
            pages.extend(aray)
    #        next=driver.find_element_by_id("opt_unusual_volume_next")
            page_num+=1
            nxt=driver.find_element_by_css_selector("a.paginate_button.next")
            
            try:
                nxt_dis=driver.find_element_by_css_selector("a.paginate_button.next.disabled")
            except:
                pass
#        else:
#            driver.close()
#            return pages, mc_date

    driver.close()
    driver.quit()
    return pages, mc_date

#from mc_website, enrich, unfiltered, to tbl_mc_raw 
def unop_mc(q_date):

    print(" ------ chamerlain unop_mc starts ------")

    from R_stat import stat_VIEW
    fm_last_earn_dt=60
    repeat_days=8
    
    pages, date=get_unop_mc()
    pages=[x for x in pages if x]  #remove empty element

    df=pd.DataFrame(data=pages[1:], columns=pages[0], index=range(len(pages)-1))
    df.rename(columns={'Symbol':'ticker', 'Volume':'v_opt', 'Calls':'pct_c', \
           'Puts':'pct_p', 'Relative Volume':'v_pct', 'Price':'p',\
           '% Chg':'p_chg', 'Earnings':'earn_dt', 'Event':'event',\
           'IV':'iv', 'IV Chg':'iv_chg'}, inplace=True)
    df['v_pct']=df.v_pct.astype(float)
    df['p_chg_abs']=df.p_chg.str.replace("+","").str.replace("-","")\
        .str.replace("%","").astype(float)
    df['p_chg']=df.p_chg.str.replace("+","")\
        .str.replace("%","").astype(float)        
        
    df['iv_chg_abs']=  df.iv_chg.str.replace("+","").str.replace("-","")\
        .str.replace("%","").astype(float)
    df['iv_chg']=  df.iv_chg.str.replace("+","")\
        .str.replace("%","").astype(float)
        
    df['v_opt']=df['v_opt'].str.replace(",", "").astype(float)
    df['v_opt_avg']=df['v_opt']/df['v_pct']
    df.event_dt=df['event'].str[:10]
#Filer outdated earn_dt from mc website   
#deal with datetime type issue: earn_dt<q_date, but no more than 2mth old
    df['earn_dt'].fillna('30-Jan-2015 BMO', inplace=True)
    df['earn_dt']= pd.to_datetime(df['earn_dt'].str.split(' ').str[0])
    con_earn_dt_1= df['earn_dt']<q_date
    con_earn_dt_2= (q_date -df['earn_dt'].dt.date)<datetime.timedelta(fm_last_earn_dt)
    con_earn_dt=con_earn_dt_1 & con_earn_dt_2
    df.loc[con_earn_dt, 'earn_dt']=np.datetime64('NaT')    
    
    df.sort_values(by=['v_pct','iv_chg_abs', 'p_chg_abs'], ascending=[False, True, True]
                ,axis=0, inplace=True)

    df['pct_c']=df['pct_c'].str.replace("%","").astype(float)
    df['date']=date
# historical raw date in 2 wks to find repeat      
    date_2wk=q_date - timedelta(repeat_days)
    qry="SELECT * FROM tbl_mc_raw where date>'%s'"%date_2wk
    dh=read_sql(qry, q_date)
#Raw tbl_mc, if not stored before
    date_exist=read_sql("SELECT DISTINCT date FROM tbl_mc_raw",q_date)
    df['date']=pd.to_datetime(df['date'])
#    if date in date_exist['date'][0]:
#        print("%s in tbl_mc_raw already"%date)
#    else:
#        to_sql_append(df, 'tbl_mc_raw')
#        print("tbl_mc_raw is updated")

#mc_intel
    volume_option_ratio=3.5
    price_chg_pct=3
    iv_chg_pct=15
    call_pct=85
    
    con_v_pct=(df.v_pct.astype(float) >=volume_option_ratio)
    con_p_chg=( df.p_chg_abs <3)
    con_p_chg_maxup= (df.p_chg<=price_chg_pct) & (df.p_chg>0)
    con_p_chg_mindn= (df.p_chg)<= (0-price_chg_pct)    
    con_p_chg_maxdn=(df.p_chg> -price_chg_pct)&(df.p_chg<0)
    
    con_iv_chg=( df.iv_chg_abs <5)
    con_iv_chg_maxup=( df.iv_chg <=iv_chg_pct) & ( df.iv_chg >=0)
    con_iv_chg_minup=( df.iv_chg >=iv_chg_pct/2)    
    con_iv_chg_mindn= df.iv_chg <= (-iv_chg_pct/2)   
    con_iv_chg_maxdn=(df.iv_chg>(-iv_chg_pct)) & (df.iv_chg<0)

    con_liq=(df.v_opt_avg >2000)
    con_p=(df.p.astype(float)<200) & (df.p.astype(float)>10)
    con_pc=(df['pct_c']>call_pct) |( df['pct_c']<(100-call_pct))
    con_call=(df['pct_c']>call_pct)
    con_put=( df['pct_c']<(100-call_pct))
    
#    catalyst=(pd.notnull(df.earn_dt)| pd.notnull(df.event))
    catalyst=(~np.isnat(df.earn_dt)) #| pd.notnull(df.event))
    
    con_to_up=con_iv_chg_maxup & con_p_chg_maxup & (con_call|con_put)& catalyst
    con_to_dn= con_iv_chg_mindn & con_p_chg_maxdn & (con_call|con_put)& catalyst
    con_to_peak=con_iv_chg_mindn & con_p_chg_maxup & con_call & (~ catalyst)
    con_to_trof= con_iv_chg_minup & con_p_chg_maxdn & (con_call|con_put) & (~ catalyst)
 #lame: complacement, long put or call for spec
    con_lame=(df.p_chg_abs >price_chg_pct) & con_iv_chg & con_v_pct 
    
    con_tag=(con_to_up |con_to_dn |con_to_peak |con_to_trof|con_lame) & con_v_pct
#    con_tag=(con_to_up|con_to_dn)
#strategy
#sweeping: con_v_pct, price not change yet, 90% call/put
#track repeating pattern
    CON_sweep =(con_v_pct & con_p_chg & con_iv_chg & con_p) & con_pc 
#    CON_liq = (con_v_pct & con_p_chg & con_iv_chg) & (con_liq & con_p) & (~CON_sweep)
##    CON=( CON_sweep | CON_liq)
    CON=CON_sweep
    du=df[CON]
    du=df[con_tag]
# FILTERD LESS TICKERS: get-si, div, earn_dt
    du['earn_dt']=pd.to_datetime(du['earn_dt'])
    for index, row in du.iterrows():
        try:
            du.loc[index,'ex_div']=get_DIV(row.ticker)[0]
        except:
            pass
        try:
            du.loc[index, 'si']=get_si_sqz(row.ticker)[0]
        except:
            pass
        try:
            du.loc[index, 'rsi']=int(float(get_rsi(row.ticker)))
        except:
            pass
#        if pd.isnull(row.earn_dt) \
#            |(((q_date -row.earn_dt.date())<datetime.timedelta(fm_last_earn_dt)) \
#            & ((q_date -row.earn_dt.date())>datetime.timedelta(0))):
#                    #        if np.isnat(du.loc[index, 'earn_dt']):
#            try:  #update with latest earn_dt 
#                du.loc[index, 'earn_dt']=get_earning_ec(row.ticker)[0]
#            except:
#                pass    
       
    rsi_mark=60
    con_rsi_mid=((du.rsi <=(100-rsi_mark)) & (du.rsi>=rsi_mark))
    con_to_up=(con_to_up &  (du.rsi<rsi_mark))
    con_to_dn= (con_to_dn &  (du.rsi>rsi_mark))
    con_to_peak=con_to_peak & (du.rsi>rsi_mark)
    con_to_trof= (con_to_trof) & (du.rsi<(100-rsi_mark))
       
    du.loc[con_to_up,'play']='bc'
    du.loc[con_to_dn,'play']='bp'   
    du.loc[con_to_peak,'play']='sc'
    du.loc[con_to_trof,'play']='sp'    
    du.loc[con_lame,'play']='lame'    
          
    if 'si' in du.columns:
        unop_show= ['ticker', 'play', 'v_pct', 'pct_c', 'iv_chg', 'p_chg', 'p', 'v_opt_avg'\
                ,'si','earn_dt', 'ex_div','rsi']
    else:
        unop_show= ['ticker','play', 'v_pct', 'pct_c', 'iv_chg', 'p_chg', 'p', 'v_opt_avg'\
                ,'earn_dt', 'ex_div','rsi']
    #to have event at last columns
    du=pd.concat([du[unop_show],du['event']], axis=1)
    candy= du.ticker.tolist()
    ds=stat_VIEW(q_date, candy)
    ds.drop('rsi', axis=1, inplace=True)  #ds has only sp500 co
    dus=pd.merge(du, ds, on='ticker', how='outer')
#Convert NaT to NaN
    dus['earn_dt']=pd.to_datetime(dus.earn_dt).dt.date
    dus.fillna('',inplace=True)
    stat_show=[ 'srtn_22_pct', 'rtn_22_pct', 'beta', \
               'fm_50', 'fm_200','fm_hi','fm_lo']  #fm_mean50 is for ticker
    show= unop_show + stat_show +['event']

#    return df, CON, con_to_up, con_to_dn, con_to_peak, con_to_trof

#    candy=ds.ticker.tolist()
#    dp=play_candy(q_date, candy, 'test')
#    play_show=['iv_30', 'iv_30_rank','iv_hv', 'hv_rank',\
#               'v_opt_pct', 'earn_date', 'ex_div', 'si']
    pd.set_option('display.expand_frame_repr', False)
  #save to tbl_mc_candy
    dusw=dus[show]

#    dusw=dusw[~dusw.index.duplicated(keep='first')]  
    print(" ----- mc_candy list ---- ")
    print(dusw)

    dusw['date']=date
#    to_sql_append(dusw, 'tbl_mc_candy')  

#find repeat ticker
    df_repeat=dh[dh.ticker.isin(dusw.ticker)]
    df_repeat.drop(['Name','event','earn_dt'], axis=1, inplace=True)
    df_repeat.sort_values(['ticker','date'], ascending=[True,False], inplace=True)
    print("   --------   today mc_candy Occured in last 10 days ------")
    print(df_repeat)
    
#    plot_base(q_date, dus[show].ticker.unique().tolist(), dus[show])
    pd.set_option('display.expand_frame_repr', True)

    return df  #unfiltered

def unop_mc_dev(q_date):
    print(" ------ chamerlain unop_mc starts ------")
    from R_stat import stat_VIEW
    fm_last_earn_dt=60
    
    pages, mc_date=get_unop_mc()
    pages=[x for x in pages if x]  #remove empty element

    df=pd.DataFrame(data=pages[1:], columns=pages[0], index=range(len(pages)-1))
    df.rename(columns={'Symbol':'ticker', 'Volume':'v_opt', 'Calls':'pct_c', \
           'Puts':'pct_p', 'Relative Volume':'v_pct', 'Price':'p',\
           '% Chg':'p_chg', 'Earnings':'earn_dt', 'Event':'event',\
           'IV':'iv', 'IV Chg':'iv_chg'}, inplace=True)
    df['v_pct']=df.v_pct.astype(float)
    df['p_chg_abs']=df.p_chg.str.replace("+","").str.replace("-","")\
        .str.replace("%","").astype(float)
    df['p_chg']=df.p_chg.str.replace("+","")\
        .str.replace("%","").astype(float)        
        
    df['iv_chg_abs']=  df.iv_chg.str.replace("+","").str.replace("-","")\
        .str.replace("%","").astype(float)
    df['iv_chg']=  df.iv_chg.str.replace("+","")\
        .str.replace("%","").astype(float)
        
    df['v_opt']=df['v_opt'].str.replace(",", "").astype(float)
    df['v_opt_avg']=df['v_opt']/df['v_pct']
#    df.event_dt=df['event'].str[:10]
    def rgx(val):
        import re
        evt=re.search('(\d{1,2}-\S{3}-\d{4})',val)
        if evt:
            return evt.group(0)
        else:
            return np.nan        
    df['event_dt']=df['event'].apply(rgx)
#Filer outdated earn_dt from mc website   
#deal with datetime type issue: earn_dt<q_date, but no more than 2mth old
#    df['earn_dt'].fillna('30-Jan-2015 BMO', inplace=True)
    df['earn_dt']= pd.to_datetime(df['earn_dt'].str.split(' ').str[0])
    df['earn_dt'].fillna('2018-01-01', inplace=True)
    df['earn_dt']=pd.to_datetime(df['earn_dt'])

    try:
        con_earn_dt_1= df['earn_dt']<q_date
        con_earn_dt_2= (q_date -df['earn_dt'].dt.date)<datetime.timedelta(fm_last_earn_dt)
        con_earn_dt=con_earn_dt_1 & con_earn_dt_2 
        df.loc[con_earn_dt, 'earn_dt']=np.datetime64('NaT')    
    except:
        print(" --con_earn_dt type error --")
        pass
    df.sort_values(by=['v_pct','iv_chg_abs', 'p_chg_abs'], ascending=[False, True, True]
                ,axis=0, inplace=True)

    df['pct_c']=df['pct_c'].str.replace("%","").astype(float)
    df['date']=q_date
#datetime.datetime.strptime(mc_date, '%Y-%m-%d').date()
    df['date']=pd.to_datetime(df['date'])
    date_exist=read_sql("SELECT DISTINCT date FROM tbl_mc_raw",q_date)
    if mc_date in date_exist['date'][0]:
        print("%s in tbl_mc_raw already"%mc_date)
    else:
        to_sql_append(df, 'tbl_mc_raw')
        print("tbl_mc_raw is updated")
    return df


def mc_intel(df_mc_raw, q_date):
#criteria
    df=df_mc_raw
    volume_option_ratio=2
    p_chg_pct_top=3
    iv_chg_pct_top=5
    call_pct=70
    put_pct=70
    liq=1000     
    
    con_iv_up_top=df['iv_chg']>iv_chg_pct_top
    con_iv_up_lo= (~con_iv_up_top) & (df['iv_chg']>0)
    con_iv_dn_top=df['iv_chg']< (0- iv_chg_pct_top)
    con_iv_dn_lo= (~ con_iv_dn_top) & (df['iv_chg']<0)
                
    con_p_up_top=df['p_chg']>p_chg_pct_top    
    con_p_up_lo= (~con_p_up_top) & (df['p_chg']>0)
    con_p_dn_top=df['p_chg']< (0-p_chg_pct_top)
    con_p_dn_lo= (~con_p_dn_top) & (df['p_chg']<0)
    
    con_c=df['pct_c'].astype(float)>call_pct
    con_p=(100-df['pct_c'].astype(float))>put_pct
                 
    con_cat=pd.notnull(df['earn_dt'])
#    con_iv_rank    

    CON_iv_up_top_p_up_top_c=con_iv_up_top & con_p_up_top & con_c & (~con_cat)  #LC->SCV, BW)
    CON_iv_up_top_p_up_lo=con_iv_up_top & con_p_up_lo & con_cat # Accumu ->LC, LCV
    CON_iv_up_top_p_dn_top_cp=con_iv_up_top & con_p_dn_top & (con_c | con_p) & (~con_cat) #LP -> BF, CAL
    CON_iv_up_top_p_dn_lo=con_iv_up_top & con_p_dn_lo & con_cat  # LP or LC- up to iv rank
    
    CON_iv_dn_top_p_up_top_cp=con_iv_dn_top & con_p_up_top & (con_c |con_p) & (con_cat) #SC/SP-> LP wait)
    CON_iv_dn_top_p_up_lo= con_iv_dn_top & con_p_up_lo & con_c & ( con_cat) #SC-> wait to LP
    CON_iv_dn_top_p_dn_top=con_iv_dn_top & con_p_dn_top & (con_cat)  #more dn ?
    CON_iv_dn_top_p_dn_lo=con_iv_dn_top & con_p_dn_lo & (con_c |con_p) & con_cat #sc/sp->LC
    
    CON_iv_up_lo_dn_p_up_top=(con_iv_up_lo | con_iv_dn_lo) & con_p_up_top & (con_c |con_p) & (~ con_cat) #sc/sp->BW, Cal
    CON_iv_up_lo_dn_p_dn_top=(con_iv_up_lo | con_iv_dn_lo) & con_p_dn_top  & con_cat #sc/sp nohope-> l=LP, LPV, more down                                              
    
    con_p=(df.p.astype(float)<200) & (df.p.astype(float)>2)
    con_liq=(df.v_opt_avg >liq)
    con_vol=df.v_pct>volume_option_ratio
    
    
# FILTER for less tickers
    df=df[con_liq & con_p & con_vol]
    
    df.loc[CON_iv_up_top_p_up_top_c,'play']='SCV, BW'
    df.loc[CON_iv_up_top_p_up_lo, 'play']='LCV'
    df.loc[CON_iv_up_top_p_dn_top_cp, 'play']='BF/CAL'
    df.loc[CON_iv_up_top_p_dn_lo, 'play']='LCP'
    
    df.loc[CON_iv_dn_top_p_up_top_cp, 'play']='LP'
    df.loc[CON_iv_dn_top_p_up_lo, 'play']='LP'
    df.loc[CON_iv_dn_top_p_dn_top, 'play']='LP'
    df.loc[CON_iv_dn_top_p_dn_lo, 'play']='LC'
    
    df.loc[CON_iv_up_lo_dn_p_up_top, 'play']='BW,CAL'
    df.loc[CON_iv_up_lo_dn_p_dn_top, 'play']='LPV'
    df_orig=df
    
    du=df[pd.notnull(df.play)]
    du['earn_dt']=pd.to_datetime(du['earn_dt'])
    for index, row in du.iterrows():
        try:
            du.loc[index,'ex_div']=get_DIV(row.ticker)[0]
        except:
            pass
        try:
            du.loc[index, 'si']=get_si_sqz(row.ticker)[0]
        except:
            pass
        try:
            du.loc[index, 'rsi']=int(float(get_rsi(row.ticker)))
        except:
            pass    
        
    if 'si' in du.columns:
        unop_show= ['ticker', 'play', 'v_pct', 'pct_c', 'iv_chg', 'p_chg', 'p', 'v_opt_avg'\
                ,'si','earn_dt', 'ex_div']
    else:
        unop_show= ['ticker','play', 'v_pct', 'pct_c', 'iv_chg', 'p_chg', 'p', 'v_opt_avg'\
                ,'earn_dt', 'ex_div']        
    du=pd.concat([du[unop_show],du['event']], axis=1)
    candy= du.ticker.tolist()
    ds=stat_VIEW(q_date, candy)
    ds.drop('rsi', axis=1, inplace=True)  #ds has only sp500 co
    dus=pd.merge(du, ds, on='ticker', how='outer')
#Convert NaT to NaN
    dus['earn_dt']=pd.to_datetime(dus.earn_dt).dt.date
    dus.fillna('',inplace=True)
    stat_show=[ 'beta','srtn_22_pct', 'rtn_22_pct',  \
               'fm_50', 'fm_200','fm_hi','fm_lo']  #fm_mean50 is for ticker
    show= unop_show + stat_show       
    pd.set_option('display.expand_frame_repr', False)
  #save to tbl_mc_candy
    dusw=dus[show]
    dusw.sort_values(by=['v_pct','pct_c'], ascending=[False,False], inplace=True)
#    dusw=dusw[~dusw.index.duplicated(keep='first')]  
    print(" ----- mc_candy list ---- ")
    print(dusw)

    dusw['date']=q_date
    to_sql_append(dusw, 'tbl_mc_candy')          
#find repeat ticker
    repeat_days=5
    date_repeat=q_date - timedelta(repeat_days)
    qry="SELECT * FROM tbl_mc_raw where date>'%s'"%date_repeat
    dh=read_sql(qry, q_date)
    
#Raw tbl_mc, if not stored before
    df_repeat=dh[dh.ticker.isin(dusw.ticker)]
    df_repeat.drop(['Name','event','earn_dt'], axis=1, inplace=True)
    df_repeat.sort_values(['ticker','date'], ascending=[True,False], inplace=True)
    print("   --------   today mc_candy Occured in last 10 days ------")
    print(df_repeat)
    
#    plot_base(q_date, dus[show].ticker.unique().tolist(), dus[show])
    pd.set_option('display.expand_frame_repr', True)

    return df  #unfiltered        
        
        
    

def download():
    import re
    import requests
    from bs4 import BeautifulSoup

    link = 'https://www.barchart.com/options/unusual-activity'

    r = requests.get(link)
    soup = BeautifulSoup(r.text, "html.parser")

    for i in soup.find_all('a', {'class': "toolbar-button download"}):
    #    print(re.search('http://.*\.apk', i.get('href')).group(0))
         print(i.get('href'))
         return i

def download_bc():
    from lxml import html
    from selenium.webdriver.chrome.options import Options
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.keys import Keys
    from selenium.common.exceptions import TimeoutException

    chrome_options=Options()
#    chrome_options.add_argument("--disable-popup")
    chrome_options.add_extension(r"c:\pycode\Github\extension_1_0_7_overlay_remove.crx")
#    chrome_options.add_extension(r"c:\pycode\Github\extension_1_13_8.crx")  #fairad
    #chrome_options.add_extension(r"G:\Trading\Trade_python\pycode\Github\extension_0_3_4.crx")
    chrome_options.add_argument('--always-authorize-plugins=true')
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--start-maximized")  #full screen
    x=r"C:\Users\jon\AppData\Local\Google\Chrome\User Data\Default"
    chrome_options.add_argument("user-data-dir=%s"%x)
    url="https://www.barchart.com/options/unusual-activity"

    gecko="c:\\pycode\Github\chromedriver.exe"
    driver=webdriver.Chrome(executable_path="c:\pycode\Github\chromedriver.exe", \
        chrome_options=chrome_options)
#LOG IN
    url_0="https://www.barchart.com/login"
    driver.get(url_0)
    time.sleep(4)
    try:
        ol=driver.find_element_by_class_name("register-flyer-close")
        ol=driver.find_element
        ol.click()
    except:
#        print("no find flyer close")
        pass
#    username = driver.find_element_by_id("bc-login-form")
#    password = driver.find_element_by_id("login-form-password")

    try:
        username=driver.find_element_by_xpath('//*[@id="bc-main-content-wrapper"]/div/div[2]/div[2]/div/div/div/div[2]/form/div[1]/input')
        password=driver.find_element_by_xpath('//*[@id="login-page-form-password"]')
        username.send_keys("boxofjon@yahoo.com")
        password.send_keys("kevin2008")
        submit=driver.find_element_by_xpath('//*[@id="bc-main-content-wrapper"]/div/div[2]/div[2]/div/div/div/div[2]/form/div[4]/button')
        time.sleep(2)
        submit.send_keys('\n')
    except:
        pass  #if user login cache preserved

    driver.get(url)
    time.sleep(4)
    download=driver.find_element_by_xpath('//*[@id="main-content-column"]/div/div[3]/div[2]/div[2]/a[4]')
    download.click()
    time.sleep(10)
#    download.send_keys(Keys.ENTER)
#    download.send_keys('\n')
    print("i am waiting")
    driver.close()
    driver.quit()

def unop_bc_dev(q_date):  #from bc.csv append to "tbl_bc_raw" , 
# https://www.barchart.com/options/unusual-activity
# q_date is used to filter today's trade only thus 
# need to be the same as date of unop file downloaded
# bc website, download, tbl_bc_raw (unfilterd)
    print("  ---- barchart unop_bc starts -----  ")
    import os
    sweep_pct=0.02
    bid_ask_mark=0.75  #mark b or s, label for agreessivness, i.e. sweeping
    vol_oi_ratio=10
    price_hi=150
    price_lo=10
    dte_min=5

    path='c:\\pycode\\eod'
#    path= 'C:\\Users\qli1\BNS_git\pycode_190106'
    path=r"%s"%path
    files=os.listdir(path)
    files_unop=[f for f in files if f[:7]=='unusual']
    if len(files_unop)==0:
        print("eod file not available")
        return
    dbc=pd.read_csv(r'c:\pycode\eod\%s'%files_unop[0])
#    dbc=pd.read_csv(r'c:\pycode\eod\unusual-options-activity-08-02-2018.csv')
    dbc.columns=dbc.columns.str.lower()
    dbc.rename(columns={'symbol':'ticker', 'exp date':'oexp_dt', 'open int':'oi'\
            ,'volume':'vol', 'vol/oi':'v_oi', 'time':'date'}, inplace=True)
#    date_fmt="{:%m/%d/%y}".format(q_date)) -not working as %s converted dt to str already!!
#    dbc=dbc[dbc.date== q_date.strftime("%m/%d/%y")] #get q_date data only
    dbc['ba_pct']= (dbc['last']-dbc['bid'])/(dbc['ask']-dbc['bid'])
    dbc.loc[dbc['ba_pct']>bid_ask_mark, 'bs']='b'
    dbc.loc[dbc['ba_pct']<(1-bid_ask_mark), 'bs']='s'
#sweeping buy or sell
#    con_sb= dbc['last']>=(dbc['ask'] * (1-sweep_pct))
#    con_ss= dbc['last']<=(dbc['bid'] *(1+sweep_pct))
    con_sweep_b=(dbc['bs']=='b')
    con_sweep_s=(dbc['bs']=='s')
#    loc_sweep_b=con_sweep_b[con_sweep_b==True].index
#    dbc.loc[con_sb, 'sweep']='b'
#    dbc.loc[con_ss, 'sweep']='s'
    dbc['sweep']=''  #empty place holder
#add volume to dbc as "stk_vol" as raw input once only
    df_bcv=dbc
    df_bcv['date']=pd.to_datetime(df_bcv['date']).dt.date
    df_pv=read_sql("SELECT * FROM tbl_pv_all wHERE date='%s'" %q_date, q_date)
    df_pv['date']=pd.to_datetime(df_pv['date']).dt.date
    df_bcv=dbc.merge(df_pv[['ticker','date','volume']], \
                     on=['ticker','date'],how='left')  
    df_bcv.rename(columns={'volume':'stk_vol'}, inplace=True)     
#    to_sql_append(df_bcv, 'tbl_bc_raw')

#FILTERED BEGIN
    con_voloi=dbc['v_oi']>vol_oi_ratio
    con_price=((dbc['price'] < price_hi) & (dbc['price']> price_lo))
    con_dte= dbc.dte>dte_min
    dbc=dbc[con_voloi & con_price & con_dte]  #dbc filtered!!
    
    dsb=dbc[con_sweep_b]
    dss=dbc[con_sweep_s]
    dsb.sort_values(['date', 'ticker','v_oi'],\
                   ascending=[False, True, False], axis=0)
    dss.sort_values(['date', 'ticker','v_oi'],\
                   ascending=[False, True, False], axis=0)
    list_dss=dss.ticker.unique().tolist()
    list_dsb=dsb.ticker.unique().tolist()
    list_sweep=list(set(list_dss + list_dsb))

#sweeping buy or sell df
    ds=pd.concat([dsb, dss], axis=0)
    ds.sort_values(['v_oi', 'dte', 'iv', 'price'], ascending=[False,False, True, True], inplace=True)
#GET p_chg info.
    p_date=q_date - datetime.timedelta(1)
    t_list=tuple(ds['ticker'].unique().tolist())
    qry="SELECT * FROM tbl_pv_all WHERE date>='%s'AND ticker in %s"%(p_date, t_list)
    dp=read_sql(qry, q_date)
    dp=dp[['ticker','date','close']]
    dp=dp.set_index(['date','ticker'])
    dp=dp.groupby(level='ticker').pct_change()
    dp.dropna(inplace=True)
    dp.columns=['p_chg']
    dp=dp.reset_index()
    ds=pd.merge(ds,dp[['ticker','p_chg']], on='ticker',how='left')
    
    
    show_bc=['ticker', 'strike', 'oexp_dt', 'dte','type', 'bs', 'p_chg'\
             , 'last', 'iv', 'price','ba_pct',  'vol', 'oi', 'v_oi', 'date','sweep']

    ds=ds[show_bc]
    pd.set_option('display.expand_frame_repr', False)
    print(" --- unop_bc filtered by sweep b/s, v_oi, dte, iv, price --- ")
    print (ds)
    pd.set_option('display.expand_frame_repr', True)
    to_sql_append(ds, "tbl_bc_bs")
#    return dbc  
    return df_bcv #return enrieched but unfiltered orig.for unop_combo()

#d0=ds.merge(df, on='ticker', how='left')
def unop_combo(dmc, dbc): #dmc_raw and dbc_(unfilterd)
    dbc.drop(['date','iv'], axis=1, inplace=True)
    do=dmc.merge(dbc, on='ticker',how='left')
    do=do[ pd.notnull(do.strike) ] 

    show_bc=['sweep', 'bs', 'type', 'strike', 'oexp_dt', 'v_oi']
    show_mc=['ticker','v_pct', 'pct_c', 'iv_chg', 'p_chg',\
             'p', 'v_opt_avg','earn_dt','iv']
    
    show_mbc=show_mc + show_bc
    show=['ticker','v_pct','v_oi','iv_chg','p_chg','p','pct_c',\
         'type','sweep','bs','strike','oexp_dt','earn_dt']
    do=do[show]
    do.sort_values(['v_pct','ticker'], ascending=[False,True], inplace=True)
#    print(do[show].sort_values)
        
#    do_both=do[con_both]  #only
    do.fillna('',inplace=True)
    pd.set_option('display.expand_frame_repr', False)
    print (" --- mc_raw + bc_filered ------ ")
    print (do)
    pd.set_option('display.expand_frame_repr', True)


    return do
#
def unop_bc_R(q_date):  #from bc.csv append to "tbl_bc_raw" , 
# https://www.barchart.com/options/unusual-activity
# q_date is used to filter today's trade only thus 
# need to be the same as date of unop file downloaded
# bc website, download, tbl_bc_raw (unfilterd)

    import os
    print("  ---- barchart unop_bc starts -----  ")
    path='c:\\pycode\\eod'
    path=r"%s"%path
    files=os.listdir(path)

    #list csv files
    files_unop=[f for f in files if f[:7]=='unusual']
    if len(files_unop)==0:
        print("eod file not available")
        return
    dbc=pd.read_csv(r'c:\pycode\eod\%s'%files_unop[0])

#    dbc=pd.read_csv(r'c:\pycode\eod\unusual-options-activity-08-02-2018.csv')


    dbc.columns=dbc.columns.str.lower()
    dbc.rename(columns={'symbol':'ticker', 'exp date':'oexp_dt', 'open int':'oi'\
            ,'volume':'vol', 'vol/oi':'v_oi', 'time':'date'}, inplace=True)
#    date_fmt="{:%m/%d/%y}".format(q_date)) -not working as %s converted dt to str already!!

#    dbc=dbc[dbc.date== q_date.strftime("%m/%d/%y")] #get q_date data only
    
    
    sweep_pct=0.02
    bid_ask_range=0.5
    vol_oi_ratio=10
    price_hi=200
    price_lo=10
    
    dbc['ba_pct']= (dbc['last']-dbc['bid'])/(dbc['ask']-dbc['bid'])
    dbc.loc[dbc['ba_pct']>bid_ask_range, 'bs']='b'
    dbc.loc[dbc['ba_pct']<bid_ask_range, 'bs']='s'

    con_voloi=dbc['v_oi']>vol_oi_ratio
    con_price=((dbc['price'] < price_hi) & (dbc['price']> price_lo))
    dbc=dbc[con_voloi & con_price]
#sweeping buy or sell

    con_sb= dbc['last']>=(dbc['ask'] * (1-sweep_pct))
    con_ss= dbc['last']<=(dbc['bid'] *(1+sweep_pct))
#    loc_sweep_b=con_sweep_b[con_sweep_b==True].index
    dbc.loc[con_sb, 'sweep']='b'
    dbc.loc[con_ss, 'sweep']='s'
#add volume to dbc as "stk_vol" as raw input once only
    df_bcv=dbc
    df_bcv['date']=pd.to_datetime(df_bcv['date']).dt.date
    df_pv=read_sql("SELECT * FROM tbl_pv_all wHERE date='%s'" %q_date, q_date)
    df_pv['date']=pd.to_datetime(df_pv['date']).dt.date
    df_bcv=dbc.merge(df_pv[['ticker','date','volume']], \
                     on=['ticker','date'],how='left')  
    df_bcv.rename(columns={'volume':'stk_vol'}, inplace=True)     
    to_sql_append(df_bcv, 'tbl_bc_raw')
#    to_sql_append(dbc, 'tbl_bc_dev')
    
    dsb=dbc[con_sb]
    dss=dbc[con_ss]
    dsb.sort_values(['date', 'ticker','v_oi'],\
                   ascending=[False, True, False], axis=0)
    dss.sort_values(['date', 'ticker','v_oi'],\
                   ascending=[False, True, False], axis=0)
    list_dss=dss.ticker.unique().tolist()
    list_dsb=dsb.ticker.unique().tolist()
    list_sweep=list(set(list_dss + list_dsb))
    show_bc=['ticker', 'price', 'type', 'strike', 'oexp_dt', 'dte', 'sweep', 'bs'\
             , 'ba_pct', 'last', 'vol', 'oi', 'v_oi', 'iv', 'date']

#    show_bc=['ticker', 'sweep', 'type', 'strike', 'exp_dt', 'dte'\
#             , 'price','v_oi', 'iv']
#sweeping buy or sell df
    ds=pd.concat([dsb, dss], axis=0)
    ds.sort_values(['ticker', 'dte'], ascending=[True,True], inplace=True)
    ds=ds[show_bc]
    pd.set_option('display.expand_frame_repr', False)
    print (ds)
    pd.set_option('display.expand_frame_repr', True)
    to_sql_append(ds, "tbl_bc_bs")
    return dbc

#d0=ds.merge(df, on='ticker', how='left')
def unop_combo(dmc, dbc): #dmc_raw and dbc_(unfilterd)
    dbc.drop(['date','iv'], axis=1, inplace=True)
    do=dmc.merge(dbc, on='ticker',how='left')
    do=do[ pd.notnull(do.strike) ] 

    show_bc=['sweep', 'bs', 'type', 'strike', 'oexp_dt', 'v_oi']
    show_mc=['ticker','v_pct', 'pct_c', 'iv_chg', 'p_chg',\
             'p', 'v_opt_avg','earn_dt','iv']
    
    show_mbc=show_mc + show_bc
    show=['ticker','v_pct','v_oi','iv_chg','p_chg','p','pct_c',\
         'type','sweep','bs','strike','oexp_dt','earn_dt']
    do=do[show]
    do.sort_values(['v_pct','ticker'], ascending=[False,True], inplace=True)
#    print(do[show].sort_values)
        
#    do_both=do[con_both]  #only
    do.fillna('',inplace=True)
    pd.set_option('display.expand_frame_repr', False)
    print (" --- mc_raw + bc_filered ------ ")
    print (do)
    pd.set_option('display.expand_frame_repr', True)
    

    return do

# generate spec_candy from tbl_mc_raw, tbl_bc_raw on q_date
def spec_candy(q_date, spec_list=[]):
    p_date=q_date-datetime.timedelta(10)
    if len(spec_list)==0:
        print("empty spec_list")
        return
    dmc=read_sql("SELECT * FROM tbl_mc_raw where date>='%s'"%p_date, q_date)
    dbc=read_sql("SELECT * FROM tbl_bc_raw where date>='%s'"%p_date, q_date)
    dmc.drop('index', axis=1, inplace=True)
    dbc.drop(['index','iv'], axis=1, inplace=True)
    dmbc=dmc.merge(dbc, on=['ticker','date'],how='outer')
    df=dmbc[dmbc.ticker.isin(spec_list)]
    df.sort_values(by=['ticker','date'], ascending=[True, False], inplace=True)
    df['note']=''
    df['lsn']=''
    df.drop_duplicates('ticker', keep='first',inplace=True)
    to_sql_append(df,"tbl_spec_candy")
    print("tbl_spec_cany appended")
    return df

def spec_track(q_date):
    trace_days=5
    date_trace=q_date - timedelta(trace_days)
    qry="SELECT * FROM tbl_spec_candy where date>'%s'"%date_trace
    df=read_sql(qry, q_date)
    show=['ticker', 'lsn', 'pct_c', 'v_pct','p', 'earn_dt', 'iv',\
        'type', 'strike','vol', 'oexp_dt', 'v_oi', 'bs', 'sweep', 'note']
    df=df[show]
    print("spec_track (5 days) plot ")
    plot_base(q_date, df.ticker.unique(),df)
    return df
    
    

def untel(q_date):
#create tbl_bc_pv to track one ticker ONLY price, volume change in next x days
#Daily replacce "tbl_bc_pv"

# prem_b_s    
#    df_mc=read_sql("SELECT * FROM tbl_mc_raw WHERE date='%s'"%q_date)
#    df_bc=read_sql("SELECT * FROM tbl_bc_raw WHERE date='%s'"%q_date)
##stk_vol confirm same day or max 3 day

    days_track=6
    p_chg=0.03
    vol_chg=2.5
    p_date=q_date-datetime.timedelta(days_track)
#Update unop_bc less than 10 days ago ONLY
    df_bc=read_sql("SELECT * FROM tbl_bc_raw wHERE date>'%s'" %p_date, q_date)
    df_pv=read_sql("SELECT * FROM tbl_pv_all wHERE date='%s'" %q_date, q_date)
    df_bc=df_bc[df_bc.dte>2]
    df_bc.drop_duplicates(['ticker','date'], keep=False, inplace=True)

    df_bc['oexp_dt']=pd.to_datetime(df_bc['oexp_dt'])
    
    df_bc['date']=pd.to_datetime(df_bc['date']).dt.date.astype(str)
    df_pv['date']=pd.to_datetime(df_pv['date']).dt.date.astype(str)


    df_bc_pv=df_bc.merge(df_pv[['ticker','close','volume']], \
            on=['ticker'], how='left')

#    df_bc_pv.rename(columns={'volume':'stk_vol'},inplace=True )
    df_bc_pv.drop('level_0', axis=1, inplace=True)
    to_sql_replace(df_bc_pv, 'tbl_bc_pv')
    df=df_bc_pv
#    df=read_sql("SELECT * FROM tbl_bc_pv", q_date)

    df['p_chg']=df['close']/df['price']-1
    df['vol_chg']=df['volume']/df['stk_vol']

    con_dt1=(q_date - pd.to_datetime(df['date']).dt.date)<datetime.timedelta(days_track)
    con_dt=( df['oexp_dt']>q_date )& con_dt1
    df=df[con_dt]

    
    con_bull=(df['type']=='Put')&(df['bs']=='s')     
    con_bull=con_bull |((df['type']=='Call')&(df['bs']=='b'))    
    con_bear=(df['type']=='Put')&(df['bs']=='b')
    con_bear=con_bear |((df['type']=='Call')&(df['bs']=='s'))
    con_up=df['p_chg']>p_chg
    con_dn=df['p_chg']<-p_chg
    con_vol=df['vol_chg']>vol_chg
    
#    df[con_bull & con_up]['conf_p']=1
#    df[con_bull & con_dn]['conf_p']= -1       
#    df[con_bear & con_dn]['conf_p']=1
#    df[con_bear & con_up]['conf_p']= -1
#    df[con_vol]['conf_v']=True
      
    df.loc[con_bull & con_up,'conf_p']=1
    df.loc[con_bull & con_dn,'conf_p']=-1          
    df.loc[con_bear & con_dn,'conf_p']=1     
    df.loc[con_bear & con_up,'conf_p']= -1
    df.loc[con_vol, 'conf_v']=True          

    con_show=pd.notnull(df['conf_p']) | pd.notnull(df['conf_v'])
    show=['ticker', 'oexp_dt', 'v_oi', 'iv', 'strike',\
          'dte','type','bs', 'sweep', 'conf_p', 'conf_v',\
          'p_chg', 'vol_chg','close', 'date']
    ds=df[con_show][show]
    ds.fillna('',inplace=True)
    ds.sort_values(['date','conf_p','sweep','bs','v_oi'], \
        ascending=[False,False,True,True,False], inplace=True)
#    df.sort_values(['date','#    
    pd.set_option('display.expand_frame_repr', False)
    print(ds)
    pd.set_option('display.expand_frame_repr',True)
    return ds


    
  

'''
1. sentiment on volume (not premium!)
   mc: vol_p, vol_c; 
   bc (new postition premium!): vol_p_b, vol_p_s, vol_c_b(), vol_c_s(bwrite)
   
2. vol_stk confirm (same day or 3 day + )
3. 
Decipline:
a. tgt_dt, tgt_p (catalyst_dt) -> give SPEC, if term=S, then out if no gain in x days
b. ibq/e/t (by term)
'''