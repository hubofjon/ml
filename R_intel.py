# -*- coding: utf-8 -*-
"""
sub_func
    1.get_DIV, (nasdaq: div_dt, yld, pe, beta)
    2.get_RSI (calc: rsi -14 days)
    3. get_SI (sqz: si, si_chg, si_dtc, cap, inst)
    3.1 get_si (alternative: nasdaq)
    4. get_EARN (whisper)

Not in use: 
    6. view_corr_etf  (180 day ETF correlation to SPY)
    7. view_rtn_etf
    8. view_corr_tsv  (22 days sp500 components correlation, sector, Vxx)
    9. view_unpv  (Today's unusual price, volume change)
    10.get_trades_marketbeat  (live trades: rating, headline, insider)
    11.marketbeat (rating, news, insider)
    12.get_ta (stocktrendchart.com    weekly trend reading)
    13.get_betas (get beta of all sp500 stocks nasdaq.com)
    14.get_share_nasdaq (nasdaq.com, ex_div, beta )
"""
import pandas as pd
import numpy as np 
import datetime as datetime
#from yahoo_finance import Share
from pandas_datareader import data as pdr
from timeit import default_timer as timer
import bs4
import requests
import time
import warnings
warnings.filterwarnings("ignore")
from termcolor import colored, cprint
from P_commons import to_sql_append, to_sql_replace, read_sql, type_convert
from pandas.tseries.offsets import BDay

today=datetime.datetime.today()
todate=today.date()
LBdate=(todate-BDay(1)).date()
#DF_sp500=pd.read_csv('c:\\pycode\pyprod\constituents.csv')
#df_etf=pd.read_csv('c:\\pycode\pyprod\etf.csv')
#df_sp500=DF_sp500.ix[:,0] #serie
#df_etf=df_etf.ix[:,0]
def get_DIV(ticker=''):
    import requests
    import bs4
    import re
    from lxml import html
    from lxml import etree
    earnings_url = 'http://www.nasdaq.com/symbol/' + ticker#.lower()
    request = requests.get(earnings_url)
    soup = bs4.BeautifulSoup(request.text, 'html.parser')
    try:    
        div_dt= soup.find(text="Ex Dividend Date").find_next('div').text.strip()
    except:
        div_dt='N/A'
    try:    
        yld= soup.find(text="Current Yield").find_next('div').text.strip()
        yld=float(yld.replace('%','').replace(' ',''))
    except:
        yld=0
    try:
        pe=soup.find(text="P/E Ratio").find_next('div').text.strip()
        pe=float(pe.replace('NE','0'))
    except:
        pe=0
    try:        
        pe_f=soup.find(text="Forward P/E (1y)").find_next('div').text.strip()
        pe=float(pe.replace('NE','0'))
    except:
        pe_f=0
    try:
        beta=soup.find(text="Beta").find_next('div').text.strip()
        beta=float(beta)
    except:
        beta=0
        #beta=soup.find(text=re.compile('beta')).findNext('td').text
#        tree = html.fromstring(request.content)
#        tbl="""//div[@class="genTable thin"]/table/tbody/"""
#        ex_div=tree.xpath(tbl+'/tr[12]/td[2]/text()')
    return div_dt, yld, pe, pe_f, beta

def get_RSI(series, period):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    rs = pd.stats.moments.ewma(u, com=period-1, adjust=False) / \
         pd.stats.moments.ewma(d, com=period-1, adjust=False)
 #https://stackoverflow.com/questions/20526414/relative-strength-index-in-python-pandas   
#    data = pd.Series( [ 44.34, 44.09, 44.15, 43.61,46.03, 46.41, 46.22, 45.64 ] )
    return 100 - 100 / (1 + rs)   

def get_SIs_sqz(df):
    for index, row in df.iterrows():
        ticker=row['ticker']
        try:
            names=['si', 'si_chg', 'pe', 'hi_1y_fm', \
                   'lo_1y_fm', 'ma_200_fm', 'ma_50_fm', 'sec','ind']
            df=pd.concat([df,pd.DataFrame(columns=names)])
            df.loc[index,names]=get_si_sqz(ticker)
            #ds=ds.append(df)
        except:
            print("error: ", ticker)
            pass
    return df

def get_SI(ticker=''):
    import urllib3
    import bs4
    import requests
    import re
    http=urllib3.PoolManager()
    url = "http://shortsqueeze.com/?symbol=" + ticker + "&submit=Short+Quote%E2%84%A2"
    request = requests.get(url)
    soup = bs4.BeautifulSoup(request.text, 'html.parser')
#    si=soup.find('div', class="Short Percent of Float")
#    tag = soup.find(text=re.compile('Short Percent of Float*'))
    ts=soup.get_text()
    ts=ts.split("\n")
    s0='Short Percent of Float'
    s1='Short % Increase / Decrease'
    s2='Short Interest Ratio (Days To Cover)'
    s3='Market Cap.'
    s4='% Owned by Institutions'
    s=[s0,s1,s2,s3]
    i0=ts.index(s0)
    si=float(ts[i0+1].strip().replace(" ","").replace("%","").replace("",'0'))/100
    si='{:.2f}'.format(si)
    
    i1=ts.index(s1)
    si_chg=float(ts[i1+1].strip().replace(" ","").replace("%",""))/100

    i2=ts.index(s2)
    si_dtc=float(ts[i2+1].strip().replace(" ",""))     
    si_dtc=float(ts[i2+1].strip().replace(" ","")) 
    
    i3=ts.index(s3)
    m_cap=float(ts[i3+2].strip().replace(" ","").replace(",",""))/1000000000
    m_cap='{:.2f}'.format(m_cap)
    
    i4=ts.index(s4)
    inst=float(ts[i4+1].strip().replace(" ","").replace("%","").replace("",'0'))/100
    inst='{:.2f}'.format(inst)
#    s=soup.find_next_siblings(tag)
#    return tag[tag.index(':') + 1:].strip()
#    si = soup.find("div", {"id": "quotes_content_left_ShortInterest1_ContentPanel"})
#    si = si.find("div", {"class": "genTable floatL"})
#    df = pd.read_html(str(si.find("table")))[0]
    return [float(si), si_chg, si_dtc, float(m_cap), float(inst) ]

def get_EARN(ticker=''):
    import urllib3
    import bs4
    import requests
    import re
    import datetime
    from dateutil import parser
    http=urllib3.PoolManager()
    url = "https://www.earningswhispers.com/stocks/" + ticker 
    request = requests.get(url)
    soup = bs4.BeautifulSoup(request.text, 'html.parser')
    earn_list=soup.find_all('div', attrs={"class":"mainitem"})
    for e in earn_list:
        if e.text !='N/A':
            earn_dt=e.text
        else:
            pass
    earn_dt=earn_dt + ' 2019'
    earn_dt=parser.parse(earn_dt)
    return earn_dt
#    earn_dt=earning.next_sibling.get_text()
#    earning_time=soup.find('div', id="earningstime").get_text()
#    st=soup.find('div', id="stprice")
#    st=st['class'][2].split("-")[1]
#    it=soup.find('div', id="itprice")['class'][2].split("-")[1]
#    lt=soup.find('div', id="ltprice")['class'][2].split("-")[1]    
#    i_sent=soup.find('div', id="vsent")['class'][2].split("-")[1]
#    a_sent=soup.find('div', id="asent")['class'][2].split("-")[1]  
    com=soup.find('div', attrs={"class":"lowerboxcont"}).get_text()     
#    return [earning, earning_time, st,it,lt,i_sent,a_sent,com]
#add Year to earning date
    today=datetime.datetime.today()
    earn=datetime.datetime.strptime(earn_dt, '%b %d' )
    if earn.month < today.month:
        earn_year=today.year+1
    else:
        earn_year=today.year
    earn_dt=datetime.datetime(earn_year,earn.month,earn.day).date()
#    return [earn_dt, earning_time, st,it,lt,i_sent,a_sent,com]
    return [earn_dt,com]

def get_earnings_ec(df):
    for index, row in df.iterrows():
        ticker=row['ticker']
        try:
            df.loc[index,'earn_dt']=get_earning_ec(ticker)
        except:
            print("get_earning_ec error: ", ticker)
            pass
    return df  

def get_mkt_profile():
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.dates import num2date
    from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MonthLocator, MONDAY, FRIDAY
    import matplotlib.ticker as mticker
    
    ds=read_sql("select * from tbl_stat_etf")
    dsp=read_sql("select * from tbl_stat")
    for df in [ds, dsp]:
        pvt=df.pivot_table(index=['date'],values=['fm_50','fm_200','fm_hi',\
                'fm_lo','rsi'], aggfunc={'fm_50': lambda x: np.sum(x>=0.0)/len(x),\
                'fm_200': lambda x: np.sum(x>=0.0)/len(x), 'fm_hi': lambda x: \
                np.sum(np.abs(x)<0.2)/len(x), 'fm_lo': lambda x: np.sum(x<=0.2)/len(x),\
                'rsi': lambda x: np.sum(x>70)/len(x)})
#        pvt[['fm_50','fm_200','fm_hi','rsi']].plot\
#        ( title="profiles: %s"%df.shape[0], alpha=0.7, rot=45)
        fig, ax = plt.subplots(figsize=(16,4))
        ax=plt.gca()
        pvt['date']=pvt.index
        pvt['date']=pd.to_datetime(pvt['date'],format='%Y-%m-%d')
        ax_date=pvt['date'].map(mdates.date2num)  
        ax.plot(ax_date,  pvt['fm_50'].tolist(), 'b-', label='abov_ma50%', alpha=0.5)
        ax.plot(ax_date, pvt['fm_200'].tolist(), 'c-', label='abov_ma200%', alpha=0.5)
        ax.plot(ax_date, pvt['fm_hi'].tolist(), 'y-', label='in_20%_to_hi%', alpha=0.5)
        ax.plot(ax_date, pvt['rsi'].tolist(), 'm-', label='rsi>70%', alpha=0.5)
        xt = ax.get_xticks()
        ax.xaxis.set_minor_locator(WeekdayLocator(FRIDAY))
        ax.xaxis.set_minor_formatter(DateFormatter('%d'))
        new_xticks = [datetime.date.isoformat(num2date(d)) for d in xt]
        ax.set_xticklabels(new_xticks,rotation=45, horizontalalignment='right')
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper left')
        plt.show()
#        ax.plot(ax_date, ax_ma_20, 'y-', label='ma_20', alpha=0.5)
#    ax.plot(ax_date, ax_ma_50, 'c-', label='ma_50', alpha=0.5)
#    plt.legend(loc="center left")
    
    
        
def get_si(ticker=''):
    import urllib3
    import bs4
    import requests
    
    http=urllib3.PoolManager()
    url = "http://www.nasdaq.com/symbol/" + ticker + "/short-interest"
#    res = urllib3.urlopen(url)
#    res = res.read()
    res=http.request('GET', url)
    html=res.read()
    soup =bs4.BeautifulSoup(html)
    si = soup.find("div", {"id": "quotes_content_left_ShortInterest1_ContentPanel"})
    si = si.find("div", {"class": "genTable floatL"})
    df = pd.read_html(str(si.find("table")))[0]
    df.index = pd.to_datetime(df['Settlement Date'])
    del df['Settlement Date']
    df.columns = ['ShortInterest', 'ADV', 'D2C']
    return df.sort()
   
def plot_corr_etf():
    qry="SELECT * FROM tbl_price_etf"
    df=read_sql(qry, todate)
    df=df[~df.index.duplicated(keep='first')]  
    df=df.set_index('date')
    df.sort_index(axis=0)
    df_risk=df[['SPY','GLD','JJC','WYDE','HYG',\
        'TLT', 'UUP']]
    risk_base=df_risk.corr().ix['SPY',:]
    dr=df_risk.tail(180)
    drp=pd.rolling_corr(dr['SPY'], dr, window=66, pairwise=True)
    drp=drp.drop(['SPY'],1)
    #plot chart
 # forex
    df_fx=df[['UUP','FXC','FXY','FXA','FXE','GLD','TLT']]
    dx=df_fx.tail(180)
    dxp=pd.rolling_corr(dr['UUP'], dx, window=66, pairwise=True)
    dxp=dxp.drop(['UUP'],1)    
    print ("180 day correlation to SPY")
    
    df_sec=df[['SPY','XLY','XLE','XLF','XLV','XLI','XLB','XLK','XLU','XLP','XRT', 'XHB']]

    plot(drp)
    plot(dxp)
    
def plot_corr_tsv():  #sp500 component correlation, sector, volatility
    import matplotlib.pyplot as plt    
    duration=200    
    lookback=22
    #get sp500 componetn correaltion
    qry="SELECT * FROM tbl_price"
    df=read_sql(qry, todate)
    df=df.set_index('date')
    df.sort_index(axis=0)  # 
    df_cosp=df.tail(duration)
    
    df_cs=pd.rolling_corr(df_cosp['SPY'], df_cosp, window=lookback, pairwise=True)
    df_cs=df_cs.tail(duration-lookback)
    df_cs.fillna(0, inplace=True)
    dc=df_cs.mean(axis=1)  #series, correlation of sp500 component
#clean data value and duplicated index
    dc.replace(np.inf, np.nan, inplace=True)
    dc.replace(-np.inf, np.nan, inplace=True)    
    dc.fillna(method='ffill', inplace=True)
    dc=dc[~dc.index.duplicated(keep='first')]
    #get sector corr
    query="SELECT * FROM tbl_price_etf"
    de=read_sql(query,todate)
    de=de.set_index('date')
    de.sort_index(axis=0)
   #get secto etf corr
    df_sec=de[['SPY','XLY','XLE','XLF','XLV','XLI','XLB','XLK','XLU','XLP']]
    df_s=pd.rolling_corr(df_sec['SPY'], df_sec, window=lookback, pairwise=True)
    df_s=df_s.tail(duration-lookback)
    df_s.fillna(0, inplace=True)
    ds=df_s.mean(axis=1) #sector ETF with SPY
    ds=ds[~ds.index.duplicated(keep='first')]
    #get vix
#    dv=de['^VIX']
    dv=de['VXX']
    dv=dv.tail(duration)
    dv=dv[~dv.index.duplicated(keep='first')]
    #get secto etf corr
    df_csv=pd.DataFrame({'cosp':dc, 'sec':ds, 'vxx':dv})   
   #plot on two axis
    fig=plt.figure()  #general figure
    plt.xticks(rotation=45) 
    plt.grid(b=True, which='major', color='black')
    ax1=fig.add_subplot(111)
    df_csv['cosp'].plot(color='b')
    #ax1.plot(df_cv['cosp'])
    df_csv['sec'].plot(color='y')
    ax2=ax1.twinx()
    #ax2.plot(df_cv['vix'])
    df_csv['vxx'].plot(color='r')
    plt.legend(loc='best')
#    plt.grid(b=True, which='minor', color='black')
    print("correlation - Blue: Sector, Brown: sp500 component")
    plt.show()

def view_rtn_etf():
    qry="SELECT * FROM tbl_stat_etf"
    df=read_sql(qry, todate)
    df=df[['ticker', 'rtn_5','rtn_22','rtn_66', 'corr','hv_22', 'hv_252','hv_m2y', \
    'mean_510', 'mean_1022', 'mean_2266','mean_66252']]
    df['momt']=df.mean_510.astype(str)+df.mean_1022.astype(str)+df.mean_2266.astype(str)+df.mean_66252.astype(str)
    df=df.drop(['mean_510', 'mean_1022', 'mean_2266','mean_66252'], axis=1)
    sector=['SPY','XLY','XLE','XLF','XLV','XLI','XLB','XLK','XLU','XLP']
    con_sec=df['ticker'].isin(sector)  #is in a list
    ds=df[con_sec]
    ds=ds.sort_values(['momt','rtn_5','rtn_22'], ascending=[False,True, True])
    print("sector view:")
    return ds

def plot(df):  
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    columns=df.shape[1]
    num_plots=df.shape[1]
    colormap=plt.cm.gist_ncar
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
    for i in range(columns):
        datarow=df.iloc[:,i]
        datarow.plot(alpha=0.8)
#        plt.plot(datarow)
#    locs, labels = plt.xticks()   
#    plt.xlabel("date")
#    #plt.ylabel("win_lose")
#    labels=df.columns
#    locs=np.arange(labels.shape[0])
#   plt.xticks( locs, labels, rotation=45 )
    plt.xticks(rotation=45 )    
    plt.legend(loc='best')
    #ax.set_xticks(labels, minor=True)
    #print (locs)
    plt.grid(b=True, which='major', color='black')
    plt.show()    
    
def view_unpv(underlying,LBdate):  #scan today unusal volume or price change
    import pandas_datareader.data as web
    v_fac=2.5 # avg. volume times
    p_fac=4/100  #price percent change
    list_all=[]
    list_tmp=[]
    da=pd.DataFrame()
    df=pd.DataFrame()
    dpv=pd.DataFrame()
    if underlying== 'sp500':
        df_symbol= df_sp500
    elif underlying == 'etf':
        df_symbol= df_etf
    len= df_symbol.shape[0]
    i=0
    while (i<=len):
        df_symbol=df_symbol[i:i+50]
        for ticker in df_symbol:
    #ticker, avg_vol, volume, DaysHigh, DaysLow, div_yield, ex_div,pe, si, peg,p_pct,p_1y, ma_50,\
    #ma_200,pct_50_ma, pct_200_ma, pct_hi, pct_lo
#            x=Share(ticker)._fetch()
#            list_tmp=[ticker,x['LastTradePriceOnly'],x['DaysRange'] ,\
#            x['DividendYield'],x['ExDividendDate'],x['PERatio'],x['ShortRatio'],\
#            x['PEGRatio'], x['OneyrTargetPrice'],x['FiftydayMovingAverage'],\
#            x['TwoHundreddayMovingAverage'], x['PercentChangeFromFiftydayMovingAverage'],\
#            x['PercentChangeFromTwoHundreddayMovingAverage'],  x['PercebtChangeFromYearHigh'],\
#            x['PercentChangeFromYearLow'],x['AverageDailyVolume'],x['Volume'], x['ChangeinPercent']
#            ]
            p= web.get_quote_yahoo(ticker)['last'][0]
            si=web.get_quote_yahoo(ticker)['short_ratio'][0]
            p_pct=web.get_quote_yahoo(ticker)['change_pct'][0]
            vol, vol_avg, ex_div, beta=1,1,1,1
            try:
                vol, vol_avg, ex_div, beta= get_share_nasdaq(ticker)
            except:
                pass
            list_tmp=[ticker, p, si, p_pct, vol, vol_avg]
            list_all.append(list_tmp)
        
        names=['ticker', 'p','si', 'p_pct', 'vol','vol_avg']
#        names=['ticker', 'close','range', 'div_yield', 'ex_div','pe', 'si', \
#            'peg','p_1y', 'ma_50', 'ma_200','pct_50_ma', 'pct_200_ma', 'pct_hi',\
#            'pct_lo', 'avg_vol','volume','p_pct']
        data=list_all
        da=pd.DataFrame(data, columns=names)
        df=df.append(da)
        i+=51
        
    df['v_pct']=df['vol'].astype(float)/df['vol_avg'].astype(float)
    df['date']=LBdate
    df.set_index('date', inplace=True) 
    con_v=np.abs(df['v_pct'])>=v_fac
    con_p=np.abs(df['p_pct'].str.strip('+-%').astype(float))>=p_fac*100
    dpv=df[con_v | con_p]
    dpv=dpv[~dpv.index.duplicated(keep='first')] 
 #   dpv=dpv.drop(['ma_50', 'ma_200','avg_vol','volume'],1)
    print(colored("unpv", 'red', 'on_cyan'))
    print(colored(dpv, 'blue'))
    

def get_trades_marketbeat():
    q_trade="SELECT * FROM tbl_trade WHERE exit_date ='N'"
    dt=read_sql(q_trade, todate)
    dt=dt.fillna(0)
    dn=pd.DataFrame(data=[], index=dt.index, columns=[['ticker','date','rating', 'p_target',\
        'rating_1', 'rating_2','news_1', 'news_2', 'news_3','ins', 'ins_1', 'ins_2']])
    dn['ticker']=dt['ticker']
    dn['date']=dt['date']

    for index, row in dn.iterrows():
        ticker=row['ticker']
        try:
            dn.loc[index, ['rating', 'p_target', 'rating_1', 'rating_2', \
            'news_1', 'news_2', 'news_3','ins', 'ins_1', 'ins_2']]=marketbeat(ticker)
        except:
            print("get_analyst error : ", ticker)
            pass
        time.sleep(2)
    du=dn.transpose()    
    len=du.shape[1]
    pd.options.display.max_colwidth = 200
    du.style.set_properties(**{'text-align': 'left'})
    for c in range(0,len):
        print(du.loc[:,c])    
        print ("   ")

def marketbeat(ticker=''):
#ref: http://yizeng.me/2014/04/08/get-text-from-hidden-elements-using-selenium-webdriver/ 
    from lxml import html
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.keys import Keys
    from selenium.common.exceptions import TimeoutException   
    from selenium.webdriver.chrome.options import Options
    chrome_options=Options()
    chrome_options.add_argument("--disable-popup")
    chrome_options.add_extension(r"c:\pycode\Github\extension_1_0_7_overlay_remove.crx")
    chrome_options.add_extension(r"c:\pycode\Github\extension_1_13_8.crx")  #fairad
    #chrome_options.add_extension(r"G:\Trading\Trade_python\pycode\Github\extension_0_3_4.crx")
    
    chrome_options.add_argument('--always-authorize-plugins=true')
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--start-maximized")  #full screen
    url="https://www.marketbeat.com/"
    gecko="c:\\pycode\Github\chromedriver.exe"
    driver=webdriver.Chrome(executable_path="c:\pycode\Github\chromedriver.exe", \
        chrome_options=chrome_options)
    driver.get(url)
#    driver.execute_script("window.open(url);")
    try:
        ol_close=driver.find_element_by_class_name("overlay-close")
        ol.click()
    except:
        pass
    time.sleep(2)
    symbol=driver.find_element_by_xpath('//input[@class="main-searchbox autocomplete ui-autocomplete-input"]')
    symbol.send_keys(ticker)
    submit=driver.find_element_by_xpath('//a[@class="main-search-button"]')
    submit.click()
    
    time.sleep(2)   #get_attribute('innerHTML')
    rating=driver.find_elements_by_xpath('//*[@id="AnalystRatings"]/div[2]/table/tbody/tr[2]/td[2]')[0].get_attribute('textContent')
    p_target= driver.find_elements_by_xpath('//*[@id="AnalystRatings"]/div[2]/table/tbody/tr[3]/td[2]')[0].get_attribute('textContent')   
    rating_1=driver.find_elements_by_xpath('//*[@id="DataTables_Table_0"]/tbody/tr[1]')[0].get_attribute('textContent')
    rating_2=driver.find_elements_by_xpath('//*[@id="DataTables_Table_0"]/tbody/tr[2]')[0].get_attribute('textContent')
   
    news_1=driver.find_elements_by_xpath('//*[@id="dvHeadlines"]/table/tbody/tr[1]/td[2]')[0].get_attribute('textContent')
    news_2=driver.find_elements_by_xpath('//*[@id="dvHeadlines"]/table/tbody/tr[2]/td[2]')[0].get_attribute('textContent')
    news_3=driver.find_elements_by_xpath('//*[@id="dvHeadlines"]/table/tbody/tr[3]/td[2]')[0].get_attribute('textContent')

    ins=driver.find_elements_by_xpath('//*[@id="InsiderTrades"]/div[1]')[0].get_attribute('textContent')
    ins_1=driver.find_elements_by_xpath('//*[@id="DataTables_Table_3"]/tbody/tr[1]')[0].get_attribute('textContent')
    ins_2=driver.find_elements_by_xpath('//*[@id="DataTables_Table_3"]/tbody/tr[2]')[0].get_attribute('textContent')
    return rating, p_target, rating_1, rating_2, news_1, news_2, news_3,ins, ins_1, ins_2
    
def get_share_nasdaq(ticker=''):
    """
    This function gets the share price for the given ticker symbol. It performs a request to the
    nasdaq url and parses the response to find the share price.
    :param ticker: The stock symbol/ticker to use for the lookup
    :return: String containing the earnings date
    http://www.ianhopkinson.org.uk/2015/11/parsing-xml-and-html-using-xpath-and-lxml-in-python/
    """
    import requests
    import bs4
    from lxml import html
    from lxml import etree
    try:
        earnings_url = 'http://www.nasdaq.com/symbol/' + ticker[0]#.lower()
        request = requests.get(earnings_url)

        tree = html.fromstring(request.content)
        tbl="""//div[@class="genTable thin"]/table/tbody/"""
        ex_div=tree.xpath(tbl+'/tr[12]/td[2]/text()')
#        beta=tree.xpath(tbl+'/tr[15]/td[2]/text()')[0]
#        key= '%s_Volume'%(ticker[0])
#        vol=tree.xpath('//label[@id="%s"]/text()'%key)[0]
#        vol_avg=tree.xpath(tbl + '/tr[4]/td[2]/text()') [0]
#        vol=float(vol.replace(",",""))
#        vol_avg=float(vol_avg.replace(",",""))
#        beta=float(beta.replace(",",""))
#        x=tree.xpath('//label[@id="%s"]/'%key).getsibling()
#        return vol , vol_avg, ex_div, beta
        return ex_div
    except:
        return 'No Data Found'
    
def get_o(ticker=''):
    """
    http://www.ianhopkinson.org.uk/2015/11/parsing-xml-and-html-using-xpath-and-lxml-in-python/
    """
    import requests
    import re
    import bs4
    from lxml import html
    from lxml import etree
    try:
        earnings_url = 'https://marketchameleon.com/Overview/' + ticker#.lower()
        request = requests.get(earnings_url)
       
        soup = bs4.BeautifulSoup(request.text, 'html.parser')
        close=soup.find('p', class_="symov_current_price").text
        p_chg=soup.find('p', class_="symov_current_pricechange").text
     #   p_chg=soup.find(text=re.compile('symov_current_pricechange num_neg')).findNext('p').text
        v_stk=soup.find(text=re.compile('Equity')).findNext('span').text
        v_stk_avg=soup.find(text=re.compile('90-Day Avg')).findNext('span').text
        v_opt=soup.find(text=re.compile('Option:')).findNext('span').text

        iv_30=soup.find(text=re.compile('Todays Stock Vol:')).findNext('span').text
        iv_rank=soup.find(text=re.compile('IV Pct Rank:')).findNext('span').text
        hv_20=soup.find(text=re.compile('20-Day')).findNext('span').text
        hv_252=soup.find(text=re.compile('52-Week \(HV')).findNext('span').text     
        div_date=soup.find(text=re.compile('Dividend:')).findNext('span').text
        div_yld=soup.find(text=re.compile('Div. Yield:')).findNext('span').text
        earning=soup.find(text=re.compile('Earnings:')).findNext('span').text
        pe=soup.find(text=re.compile('P/E Ratio:')).findNext('span').text      
        cap=soup.find(text=re.compile('Market Cap:')).findNext('span').text
#        tree = html.fromstring(request.content)
#        tbl="""//div[@class="genTable thin"]/table/tbody/"""
#        ex_div=tree.xpath(tbl+'/tr[12]/td[2]text()')
#        beta=tree.xpath(tbl+'/tr[15]/td[2]/text()')[0]
#        key= '%s_Volume'%(ticker[0])
#        vol=tree.xpath('//label[@id="%s"]/text()'%key)[0]
##        vol_avg=tree.xpath(tbl + '/tr[4]/td[2]/text()') [0]
##        vol=float(vol.replace(",",""))
##        vol_avg=float(vol_avg.replace(",",""))
##        beta=float(beta.replace(",",""))
#        x=tree.xpath('//label[@id="%s"]/'%key).getsibling()
##        return vol , vol_avg, ex_div, beta
#        for tr in soup.find_all('tr'):
#            tds=soup.find_all('td')
#            print(tds[0].text, tds[1].text)
        
        #return soup
        values=[ticker, close, p_chg, v_stk, v_stk_avg, v_opt, iv_30, iv_rank,\
            hv_20, hv_252, div_date, div_yld, earning, pe, cap]
    except:
        value=[ticker, 1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        pass
    names=['ticker', 'close', 'p_chg', 'v_stk', 'v_stk_avg', 'v_opt', 'iv_30', 'iv_rank',\
            'hv_20', 'hv_252', 'div_date', 'div_yld', 'earning', 'pe', 'cap']
    do=pd.DataFrame(data=[], columns=names)    
    do.loc[0,names]=values
    return do
