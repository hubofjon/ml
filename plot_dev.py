# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 16:11:51 2019

@author: qli1
"""
ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
         'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
         'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
         'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
         'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(ipl_data)

def p1():
    import matplotlib.ticker as ticker
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd
    import pandas.io.sql as pd_sql
    
    conn=sq3.connect(r'c:\Users\qli1\BNS_wspace\flask\f_trd\trd.db')
    df=pd_sql.read_sql("SELECT * FROM tbl_price_etf", conn)
    conn.close()
    df=df[['date','SPY']]
    df['date'] = pd.to_datetime(df['date'])#, unit='s')
    df['mdate'] = [mdates.date2num(d) for d in df['date']]
    df.set_index('date')

    fig, ax = plt.subplots()
    ax.plot(df['SPY'],'-')
    
#    formatter = ticker.FormatStrFormatter('$%1f')
#    ax.yaxis.set_major_formatter(formatter)
#    daysFmt = mdates.DateFormatter("'%d")
#    days = mdates.DayLocator() 
#    ax.xaxis.set_major_locator(mdates.YearLocator())
#    ax.xaxis.set_minor_locator(mdates.MonthLocator())
#  
#    yearFmt = mdates.DateFormatter("'%y")
#    ax.xaxis.set_major_formatter(yearFmt)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=60)
    
#    for tick in ax.yaxis.get_major_ticks():
#        tick.label1On = False
#        tick.label2On = True
#        tick.label2.set_color('green')
    plt.legend(loc='best')
    plt.xlabel('date')
    plt.ylabel('value')
    plt.title('demo')
    plt.tight_layout()
#    ax.set_xlime(0,10)
    plt.show()
    
def p2():
    fig5 = plt.figure(constrained_layout=True)
    widths = [2, 3, 1.5]
    heights = [1, 3, 2]
    spec5 = fig5.add_gridspec(ncols=3, nrows=3, width_ratios=widths,
                              height_ratios=heights)
    for row in range(3):
        for col in range(3):
            ax = fig5.add_subplot(spec5[row, col])
            label = 'Width: {}\nHeight: {}'.format(widths[col], heights[row])
            ax.annotate(label, (0.1, 0.5), xycoords='axes fraction', va='center')
    
def p3():
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    fig6, f6_axes = plt.subplots(ncols=3, nrows=3, constrained_layout=True,
            gridspec_kw=gs_kw)
    for r, row in enumerate(f6_axes):
        for c, ax in enumerate(row):
            label = 'Width: {}\nHeight: {}'.format(widths[c], heights[r])
            ax.annotate(label, (0.1, 0.5), xycoords='axes fraction', va='center')
            
            

def p4_blend():
    import matplotlib.transforms as transforms
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots()
    x = np.random.randn(1000)
    
    ax.hist(x, 30)
    ax.set_title(r'$\sigma=1 \/ \dots \/ \sigma=2$', fontsize=16)
    
    # the x coords of this transformation are data, and the
    # y coord are axes
    trans = transforms.blended_transform_factory(
        ax.transData, ax.transAxes)
    
    # highlight the 1..2 stddev region with a span.
    # We want x to be in data coordinates and y to
    # span from 0..1 in axes coords
    rect1 = mpatches.Rectangle((-2, 0), width=1, height=1,
                             transform=trans, color='orange',
                             alpha=0.5)
    rect2 = mpatches.Rectangle((1, 0), width=1, height=1,
                             transform=trans, color='yellow',
                             alpha=0.5)    
    ax.add_patch(rect1)
    ax.add_patch(rect2)    
    plt.show()

'''
date:
https://matplotlib.org/api/dates_api.html?highlight=num2#date-tickers
blended transformation:
https://matplotlib.org/tutorials/advanced/transforms_tutorial.html#sphx-glr-tutorials-advanced-transforms-tutorial-py
fig layout:
https://matplotlib.org/tutorials/intermediate/gridspec.html#sphx-glr-tutorials-intermediate-gridspec-py
tutorial
https://pythonprogramming.net/bar-chart-histogram-matplotlib-tutorial/?completed=/legends-titles-labels-matplotlib-tutorial/
ohlc_date sample:
https://stackoverflow.com/questions/44951178/matplotlib-dates-in-datetime-format

'''