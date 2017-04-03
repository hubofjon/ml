
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pandas.io.data as web
import datetime
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plot
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import scipy as sp
import pickle
from sklearn import neighbors

warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.2f}'.format
#need label be numeric
def ml_linear(df):

    lr = linear_model.LinearRegression()
       #y=y.applymap(lambda x:1 if x else 0)
  #split dataset
    
    #df.rename(columns={'rtn_22_fwd_pct':'zrtn_22_fwd_pct'}, inplace=True)  
    df.sort_index(axis=1)
    x=df.iloc[:,:-1]
    y=df.iloc[:,df.shape[1]-1]
    x=x.values
    x=pre_process(x)
    y=y.values
    #print (x.shape, y.shape)
    x_train, x_test, y_train, y_test= train_test_split(x,y, train_size = 0.8)
    #print (x_train.shape, y_train.shape)
#standardize features
    scaler= preprocessing.StandardScaler().fit(x_train)    
    x_train=scaler.transform(x_train)
    x_test=scaler.transform(x_test)
    
    lr.fit(x_train, y_train)

    y_test_pred=lr.predict(x_test)
    error=y_test-y_test_pred
    
    import sklearn.metrics as sm
    print ("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2) )
    return y_test_pred, y_test

def ml_knn(df): #knn requires target be lable (not numeric)
    from sklearn import neighbors
    clf=neighbors.KNeighborsClassifier(n_neighbors=3)  #n=3
    df.sort_index(axis=1)
    
    x=df.iloc[:,:-1]
    y=df.iloc[:,df.shape[1]-1]
#dataframe indexing: df[train_index] will return columns labels rather than row indices
#ndarray indexing: collection of row indicies
#so either use .iloc accessor or conver dataframe to array, x=x.values
    x=x.values 
    y=y.values
    
    from sklearn.cross_validation import KFold
    scores=[]
    kf=KFold(n=len(x), n_folds=5, shuffle=False, random_state=None)
    for train_index, test_index in kf:
        x_train, y_train=x[train_index],y[train_index]
        #print("train:", train_index, "test:", test_index)
        x_test, y_test=x[test_index], y[test_index]
        clf.fit(x,y)
        scores.append(clf.score(x_test, y_test))
    print ("mean (scores) =%.5f\t Stddev(scores)=%.5f"%(np.mean(scores),np.std(scores)))
# dave the model
    pickle.dump(clf, open(r'G:\\Trading\Trade_python\pycode\pytest\ml_knn_model.dat',"wb"))
    #clf_1=pickle.load(open(r'G:\\Trading\Trade_python\pycode\pytest\ml_knn_model.dat',"r") )
#ml_knn_model_1.dat:  ['hi_5y', 'hi_10y', 'hi_22y', 'hi_66y', 'lo_5y','lo_10y', 'lo_22y',\
#     'lo_66y']
    
def ml_knn_apply(df):
    from sklearn import neighbors
    df.sort_index(axis=1)
    x=df.iloc[:,:-1]
    y=df.iloc[:,df.shape[1]-1]
    x=x.values 
    y=y.values
    clf=pickle.load(open(r'G:\\Trading\Trade_python\pycode\pytest\ml_knn_model_1.dat',"rb"))
    y_pred=clf.predict(x)
    #error=y-y_pred
   # dc=pd.DataFrame([y_pred, y], columns=['y_pred','y'])
    ar=np.concatenate((y_pred, y), axis=0)   
    t=y_pred==y
    return t

def ml_knn_apply_1(df):
    from sklearn import neighbors
#    df.sort_index(axis=1)
#    x=df.iloc[:,:-1]
#    y=df.iloc[:,df.shape[1]-1]
#    x=x.values 
#    y=y.values
    x=df.values
    clf=pickle.load(open(r'G:\\Trading\Trade_python\pycode\pytest\ml_knn_model_1.dat',"rb"))
    y_pred=clf.predict(x)
    dt=pd.DataFrame(data=[x,y_pred])
    #error=y-y_pred
   # dc=pd.DataFrame([y_pred, y], columns=['y_pred','y'])
#    ar=np.concatenate((y_pred, y), axis=0)   
#    t=y_pred==y
    return dt
    
def ml_logistic(df):
    from sklearn.linear_model import LogisticRegression
    clf=LogisticRegression()
    df.sort_index(axis=1)
    
    x=df.iloc[:,:-1]
    y=df.iloc[:,df.shape[1]-1]
    x=x.values
    y=y.values

def ml_polyfit(df):
    df.sort_index(axis=1)
    y=df.iloc[:,df.shape[1]-1]
    for i in range(9):
        x=df.iloc[:,i]
        
#    fp1, residuals, rank, sv, rcond=sp.polyfit(x,y,1,full=True)
#    print ("model parameters: %s" % fp1)
        plt.scatter(x,y)
        plt.autoscale(tight=True)
        plt.grid()
        print("feature -- %s" %i)
        plt.show()
#d3=d2[d2.z>=0.8]
#ml_polyfit(d3) 
#   
def fit_result(y_test, y_test_pred):
    import matplotlib.pyplot as plt
    f=plt.figure(figsize=(7,5))
    ax=f.add_subplot(111)
    ax.hist(y_test-y_test_pred, bins=50)
    ax.set_title("Histogram of Residuals")
    
    from scipy.stats import probplot
    probplot(y_test- y_test_pred, plot=ax)
    plt.show()
#df=pd.read_excel(open(r'C:\Users\liq6\pycode\\test_Oct17', sheetname='sheet1'))


def thresh(df):
    #feature
    ds=pd.DataFrame()
    dr=pd.DataFrame()
    df_by_fi=pd.DataFrame()
    df_all_fi=pd.DataFrame()
    f=df.iloc[:,:10]
    #target
    L=df['z']
    gate_t=0.2   # L: >gate, S: <gate
    gate_ratio=0.2
    top=L<=gate_t # L or S
    
    best_acc_u=-1.0
    best_acc_d=-1.0
    best_acc_b=-1.0
    len=f.shape[1]

    for fi in range(len):
        thresh=f.iloc[:, fi].copy()
        thresh.sort()
        #beyond a thresh
        ds['fi_cnt']=fi
        for tu in thresh:
            pred=(f.iloc[:,fi]>tu)
            acc=(pred==top).mean()
            num=f[f.iloc[:,fi]>tu].shape[0]
            ratio=num/(f.shape[0]*(1-gate_t))
            
            if (acc> best_acc_u) & (ratio>=gate_ratio):
                best_acc_u=acc
                best_fu=fi
                best_tu=tu
                best_nu=num
                best_ru=ratio
                up_hit= df[(df.iloc[:,best_fu]>best_tu) & (df.z<=gate_t)].shape[0]
                up_hit_ratio=up_hit/best_nu
        ds['fu']=list(f)[best_fu]
        ds['fu_seq']=best_fu
        ds['acc_u']=best_acc_u
        ds['tu']=best_tu
        ds['up_fetched']=best_nu
        ds['up_hit_cnt']= up_hit
        ds['up_hit_ratio']=up_hit_ratio
        ds['up_hit']=df[(df.iloc[:,best_fu]>best_tu) & (df.z<=gate_t)].z
  
    #below a thresh
        for td in thresh:
            pred=(f.iloc[:,fi]<td)
            acc=(pred==top).mean()
            num=f[f.iloc[:,fi]<td].shape[0]
            ratio=num/(f.shape[0]*(1-gate_t))
            
            if (acc> best_acc_d)& (ratio>=gate_ratio):
                best_acc_d=acc
                best_fd=fi
                best_td=td
                best_nd=num  #number of fetched result
                best_rd=ratio
                dn_hit= df[(df.iloc[:,best_fd]<best_td) & (df.z<=gate_t)].shape[0]
                dn_hit_ratio=dn_hit/best_nd               
                
        ds['fd']=list(f)[best_fd]
        ds['fd_seq']=best_fd
        ds['acc_d']=best_acc_d
        ds['td']=best_td
        ds['dn_fetched']=best_nd
        ds['dn_hit_cnt']= dn_hit
        ds['dn_hit_ratio']=dn_hit_ratio
        ds['dn_hit']=df[(df.iloc[:,best_fd]<best_td) & (df.z<=gate_t)].z

    # in btn
        for tbu in thresh:
            for tbd in thresh:
                if tbu>tbd:
                    pred=(f.iloc[:,fi]>tbd) & (f.iloc[:,fi]<tbu)
                    acc=(pred==top).mean()
                    num=f[(f.iloc[:,fi]>tbd) & (f.iloc[:,fi]<tbu)].shape[0]
                    ratio=num/(f.shape[0]*(1-gate_t))
            
                    if (acc> best_acc_d)& (ratio>=gate_ratio):
                        best_acc_b=acc
                        best_fb=fi
                        best_tbu=tbu
                        best_tbd=tbd
                        best_nb=num
                        best_rb=ratio 
                        btn_hit= df[(df.iloc[:,best_fb]<best_tbu) & (df.iloc[:,best_fb]>best_tbd)\
                            & (df.z<=gate_t)].shape[0]
                        btn_hit_ratio=btn_hit/best_nb    

        ds['fb']=list(f)[best_fb]
        ds['fb_seq']=best_fb
        ds['acc_b']=best_acc_b
        ds['tbu']=best_tbu
        ds['tbd']=best_tbd       
        ds['btn_fetched']=best_nb
        ds['btn_hit_cnt']= btn_hit
        ds['btn_hit_ratio']=btn_hit_ratio
        ds['btn_hit']=df[(df.iloc[:,best_fb]<best_tbu) &  (df.iloc[:,best_fb]>best_tbd)\
            & (df.z<=gate_t)].z

        df_by_fi=df_by_fi.append(ds)    
#   SUMMARY OF THE BEST OF ALL FEATURE
    dr['best_fu']= list(f)[best_fu]
    dr['best_fu_seq']=best_fu
    dr['best_tu']=best_tu
    dr['up_fetched']=best_nu
    dr['up_hit_cnt']=up_hit
    dr['up_hit_ratio']=up_hit_ratio
    dr['up_hit']=df[(df.iloc[:,best_fu]>best_tu) & (df.z<=gate_t)].z     
    
    dr['fd']=list(f)[best_fd]
    dr['fd_seq']=best_fd
    dr['td']=best_td
    dr['dn_fetched']=best_nd
    dr['dn_hit_cnt']= dn_hit
    dr['dn_hit_ratio']=dn_hit_ratio
    dr['dn_hit']=df[(df.iloc[:,best_fd]<best_td) & (df.z<=gate_t)].z
    
    dr['fb']=list(f)[best_fb]
    dr['fb_seq']=best_fb
    dr['tbu']=best_tbu
    dr['tbd']=best_tbd       
    dr['btn_fetched']=best_nb
    dr['btn_hit_cnt']= btn_hit
    dr['btn_hit_ratio']=btn_hit_ratio
    dr['btn_hit']=df[(df.iloc[:,best_fb]<best_tbu) &  (df.iloc[:,best_fb]>best_tbd)\
            & (df.z<=gate_t)].z
            
#    
#    print("up:  ", list(f)[best_fu], best_fu, best_acc_u, best_tu,\
#           "num#:", best_nu, "ratio", best_ru)             
#    print("up_hit:", df[(df.iloc[:,best_fu]>best_tu) & (df.z>=gate_t)].z)
#    
#    print("dn:  ", list(f)[best_fd], best_fd, best_acc_d, best_td,\
#           "num#:", best_nd, "ratio", best_rd)
#    print("dn_hit:", df[(df.iloc[:,best_fd]<best_td) & (df.z>=gate_t)].z) 
#    print("btn:  ", list(f)[best_fb], best_fb, best_acc_b, best_tbd, best_tbu,\
#           "num#:", best_nb, "ratio", best_rb)
#           
# 
#    print("btn_hit: ", df[(df.iloc[:,best_fb]>best_tbd) & (df.iloc[:,best_fb]<best_tbu) &\
#        (df.z>=gate_t)].z)
    df_all_fi=dr
    df_by_fi.to_csv(r'G:\\Trading\Trade_python\pycode\pytest\thresh_by_fi.csv')
    df_all_fi.to_csv(r'G:\\Trading\Trade_python\pycode\pytest\thresh_all_fi.csv')    
    
    print ("thresh analysis is saved")
    
    #apply thresh to data and test result
    
    return df_by_fi, df_all_fi
#m_l(lf)
def df_enrich(df): # to be done at df_stat once stablized
    df['rtn_ma10']=df['mean_5']/df['mean_10']-1
    df['rtn_ma22']=df['mean_5']/df['mean_22']-1
#    df['rtn_ma66']=df['mean_5']/df['mean_66']-1
#    df['rtn_ma252']=df['mean_5']/df['mean_252']-1
    #ranking
    df['rtn_ma10_pct']=df['rtn_ma10'].rank(pct=True)  
    df['rtn_ma22_pct']=df['rtn_ma22'].rank(pct=True)  
#    df['rtn_ma66_pct']=df['rtn_ma66'].rank(pct=True)  
#    df['rtn_ma252_pct']=df['rtn_ma252'].rank(pct=True)
    
#    df['rtn_22_fwd']=df['p_22_fwd']/df['close_qdate']
#    df['rtn_44_fwd']=df['p_44_fwd']/df['close_qdate']
#    df['rtn_66_fwd']=df['p_66_fwd']/df['close_qdate']
#    df['rtn_22_fwd_pct']=df['rtn_22_fwd'].rank(pct=True)
#    df['rtn_44_fwd_pct']=df['rtn_44_fwd'].rank(pct=True)
#    df['rtn_66_fwd_pct']=df['rtn_66_fwd'].rank(pct=True)
#   use yearly hi or lo as base for relative price position 
    df['hi_5y']=df['hi_5']/df['hi_252']
    df['hi_10y']=df['hi_10']/df['hi_252']
    df['hi_22y']=df['hi_22']/df['hi_252']
    df['hi_66y']=df['hi_66']/df['hi_252']
    
    df['lo_5y']=df['lo_5']/df['lo_252']
    df['lo_10y']=df['lo_10']/df['lo_252']
    df['lo_22y']=df['lo_22']/df['lo_252']
    df['lo_66y']=df['lo_66']/df['lo_252']
    
    df['hv_5y']=df['hv_5']/df['hv_252']
    df['hv_10y']=df['hv_10']/df['hv_252']
    df['hv_22y']=df['hv_22']/df['hv_252']
    df['hv_66y']=df['hv_10']/df['hv_252']
    #df=select_label(df)
#remove outlier
    df=df[(df.hv_10<1)&(df.hv_22<1)&(df.hv_5<1)&(df.hv_66<1)]
    
    #drop null value
    #df=df.dropna()
    #df.isnull().sum().sum()
    return df

def filter_label(df):
#    rank_L=0.9
#    rank_S=0.1
##    df['z']=df['rtn_22_fwd'].rank(pct=True)  #use numerical value
## below is for multiple classification for KNN
# 
#    con_L= (df['rtn_22_fwd_pct']>=rank_L) & (df['rtn_22_fwd']>1) #& (df['p_44_fwd']>df['p_22_fwd'])
#    con_S= (df['rtn_22_fwd_pct']<=rank_S) & (df['rtn_22_fwd']<1) #& (df['p_44_fwd']<df['p_22_fwd'])
#    con_Z= np.abs(df['rtn_22_fwd']-1)<=0.02
#
#    df.loc[con_L, 'z']='L'   
#    df.loc[con_S, 'z']='S' 
#    df.loc[con_Z, 'z']='Z' 
    df.loc[(df.p_22_fwd-df.close_qdate)>df.p_22_sig*1, 'z22']='LL'
    df.loc[(df.close_qdate-df.p_22_fwd)>df.p_22_sig*1, 'z22']='SS'
    df.loc[np.abs(df.close_qdate/df.p_22_fwd-1)<=0.01,'z22']='ZZ'
    
    #df=df.drop(['p_22_fwd'], axis=1)
    return df #dl, ds
    
def filter_feature(df):
    df=df.drop(['Unnamed: 0', 'ticker', '25%', '50%', '75%', 'close_22b', 'close_66b',\
       'close_qdate', 'count', 'hi_10', 'hi_22', 'hi_252', 'hi_44', 'hi_5',\
       'hi_66', 'hv_10', 'hv_22', 'hv_252', 'hv_5', 'hv_66', 'lo_10', 'lo_22',\
       'lo_252', 'lo_44', 'lo_5', 'lo_66', 'max', 'mean', 'mean_10', 'mean_22',\
       'mean_252', 'mean_5', 'mean_66', 'min',\
       'rtn_22', 'rtn_5', 'rtn_66', 'std', 'date',\
       'mean_510', 'mean_1022', 'mean_2266',\
       'mean_66252', 'p_22_sig', 'p_44_sig', 'p_66_sig',\
       'rtn_5_pct',  'rtn_22_pct', 'rtn_66_pct',\
         'play', 'p_22_fwd', 'rtn_ma10', 'rtn_ma22', 'rtn_ma66', 'rtn_ma252',\
        'hi_5y', 'hi_10y', 'hi_22y', 'hi_66y', 'lo_5y', 'lo_10y', 'lo_22y',\
               'lo_66y'], axis=1)
    df=df.dropna()
    return df

def pre_process(df):
    from sklearn import preprocessing
    scaler=preprocessing.StandardScaler()
    scaler.fit(df)
    return scaler.transform(df).mean(axis=0)

  


#
###d1=ds[['sharpe_5_pct', 'sharpe_22_pct', 'sharpe_66_pct', 'p_252_pct','z']]
#d1=ds[['hi_5y', 'hi_10y', 'hi_22y', 'hi_66y', 'lo_5y','lo_10y', 'lo_22y',\
#     'lo_66y']]     
#
#d1=d1.dropna()
#d1a=d1.head(5000)

#ml_knn(d1a)

#d1b=d1.tail(1000)
#t=ml_knn_apply(d1b)
#dt=pd.DataFrame(data=t, columns=['t'])
#print (dt[dt.t==True].count()/dt.shape[0])
#dg=select_label(dg)
#dg=select_feature(dg)

#dg=df.drop([ 'hv_5', 'hv_66', 'lo_10', 'lo_22', 'lo_252', 'lo_5', 'lo_66'], axis=1)

#df=pd.read_csv(r'G:\Trading\Trade_python\pycode\test_Oct17.csv')
#df=df_enrich(df)
#df=select_feature(df)
#d2=df.head(1000)
#dff=df.head(40000)
#
#
#d2.sort_index(axis=1)
#    
#x=d2.iloc[:,:-1]
#dx=pre_process(x)

#ds,dr= thresh(d2)
# >>>>>>>>   EXECUTION ZONE    <<<<
#x=d2[(d2.rtn_ma66_pct< 0.0197)|((0.312 < d2.rtn_ma66_pct) & (d2.rtn_ma66_pct <0.332))\
#    &(d2.rtn_ma252_pct<0.66)]
#print (x.shape, x[x.z<=0.5].shape[0])
#1000 sample result
#dn: rtn_ma66_pct<0.0184 (13 out of 30)
#btn rtn_ma22_pct btn(0,91, 0.75)
#up: hv_22>0.68 (9/30)
#up & btn joint 0-> give good result
#feature group( rtn_ma66_pct, hv_10, p_66, (btn) rtn_ma10_pct/ 4 important
# (1000sample: rtn_ma66_pct, hv_22, (btn)rtn_ma22_pct)
#hv_22, rtn_ma22_pct, rtn_ma10_pct, hv_m2y_pct
#p_66, hv_66, hv_5, hv_10)
def select_test(df):
    tlf=df[(df.hv_5<0.5) & (df.hv_66<0.25) & (df.rtn_ma10_pct<0.1)]
        
    tlc=df[ (df.rtn_ma66_pct<0.1)& (df.hv_m2y_pct>0.6)]
    tsc=df[ (df.rtn_ma66_pct>0.9)  & (df.rtn_ma22_pct<0.7)\
        & (df.hv_22<0.3) & (df.hv_22>0.1)
    ]

    return tlf, tlc, tsc

    
def feature_plot(df):
    
    df.sort_index(1)
    
    import matplotlib.pyplot as plt
    #assign color based on win_22 True or False
    rows=df.shape[0]

    cols=df.shape[1]
    #pcolor="blue"
    for i in range(rows):
#       if df.iloc[i,cols-1]>=0.9:
        if df.iloc[i,cols-1]=='SS':
            pcolor="blue"
        else: 
            pcolor="yellow"
#        #plot rows of data as if they were series data
#        datarow=df.iloc[i,0:cols-1]
#        datarow.plot(color=pcolor, alpha=0.7)
        
        datarow=df.iloc[i,:cols-1]
        #datarow=datarow.astype(float)
        datarow.plot(color=pcolor, alpha=0.3)
       
    locs, labels = plt.xticks()   
    plt.xlabel("feature index")
    #plt.ylabel("win_lose")
    labels=df.columns
    
    locs=np.arange(labels.shape[0])
    plt.xticks( locs, labels, rotation=90 )
    #ax.set_xticks(labels, minor=True)
    #print (locs)
    plt.show()
    
#  >>>    EXECUTION ZONE    <<<<
#dd=data_import()
#df=dd
#df=df_enrich(df)
#df=filter_label(df)
#df=filter_feature(df)
#dff=df[['sharpe_22_pct','sharpe_66_pct','rtn_ma66_pct','rtn_ma252_pct','hv_m2y','z22']]
#ds=dff.loc[dff.z22=='SS']
#dl=dff.loc[dff.z22=='LL']
#dz=dff.loc[dff.z22=='ZZ']
#s=df.loc[df.z22=='SS']
#l=df.loc[df.z22=='LL']
#z=df.loc[df.z22=='ZZ']

#feature_plot(df)

#feature distribution

#plt.hist(d2.iloc[:,1])
#plt.hist(d2[d2.z>=0.9].iloc[:,0])

def data_import(underlying):
    if underlying=='sp500':    
        dd=pd.read_csv(r'G:\Trading\Trade_python\pycode\pytest\test_stat_sp500_20161007_20170105.csv') 
    else:
        dd=pd.read_csv(r'G:\Trading\Trade_python\pycode\pytest\test_stat_etf_6.csv')
    dd=dd.dropna()
    dd['rtn_22_fwd']=dd['p_22_fwd']/dd['close_qdate']-1
    dd['rtn_22_fwd_pct']=dd['rtn_22_fwd'].rank(pct=True)
    return dd
    
def plot_prep(df):
    do=df
    dxx=df[['hv_252_pct','rtn_ma66_pct','p_10','p_22','hv_10','hv_66',\
    'hv_5', 'hv_22', 'hv_252','rtn_22_fwd', 'rtn_22', 'rtn_66',\
    'hv_66_pct','hv_22_pct','rtn_22_pct','rtn_22_fwd_pct']]
    
    dx=df[['hv_22', 'hv_252','p_252', 'rtn_ma66_pct',  'rtn_22_pct', 'rtn_5_pct', 'hv_5',\
        'rtn_22_fwd_pct','rtn_22_fwd', 'hv_m2y', 'hv_m2y_pct']]  
    dxx['hv_5m']=dxx['hv_5']/dxx['hv_22']    
      
    df=df[['hv_22', 'hv_252','p_252', 'p_66', 'sharpe_5', 'sharpe_22', \
    'sharpe_66', 'hv_m2y', 'sharpe_22_pct',
           'rtn_22_pct', 'sharpe_66_pct', 'rtn_66_pct', 'p_252_pct', 'p_10_pct',
           'p_22_pct', 'p_66_pct', 'hv_5_pct', 'hv_22_pct', 'hv_66_pct',
           'hv_252_pct', 'hv_m2y_pct', 'rtn_ma66_pct', 'rtn_ma252_pct',\
           'rtn_22_fwd', 'rtn_22_fwd_pct']]
    df['hv_5m']=dx['hv_5']/dx['hv_22'] 
    df=dxx
    outlier=((df.hv_252.astype(float)>0.5)|(df.hv_5.astype(float)>1.1)|(df.hv_10.astype(float)>1))
    df=df[~outlier]
    df.sort_index(axis=1)
    con_z=np.abs(df.rtn_22_fwd)<=0.02
    con_l=df.rtn_22_fwd_pct>=0.8
    con_s=df.rtn_22_fwd_pct<=0.2
    dz=df[con_z]
    ds=df[con_s]
    dl=df[con_l]
    dnz=df[~con_z]
    dns=df[~con_s]
    dnl=df[~con_l]
    len=dz.shape[1]
    return do,df, dx, dl, ds, dz, dnl, dns, dnz, len

def kde_plot(len):
    for i in range(len):
        p1=dz
        p2=dnz
        p3=dz
        sns.kdeplot(p1.iloc[:,i], label="z_%s"%p1.columns.values[i])
        sns.kdeplot(p2.iloc[:,i], label="nz_%s"%p2.columns.values[i])
#        sns.kdeplot(p3.iloc[:,i], label="z_%s"%p3.columns.values[i]) 
    #    sns.kdeplot(dnz.iloc[:,i])
    #    sns.kdeplot(ds.iloc[:,i])
    #    sns.kdeplot(dl.iloc[:,i])
        plt.figure()
#dd, dl, ds, dz, dnl, dns, dnz, len=plot_prep()

#kde_plot(len)

def feature_vet(df):
    
#    fl1=df.hv_252_pct>0.5
#    fl2=df.hv_66_pct<0.6
##    fl3=df.hv_22_pct>0
#    fl4=df.rtn_ma66_pct>0.8
##    fl5=df.hv_m2y_pct>0.5
##    fl6=df.rtn_22_pct<0.8
#    
#    fs1=df.hv_252_pct>0.8
##    fs2=df.hv_22_pct>0.8  #no tnecessary
#    fs3=df.rtn_ma66_pct>0.4
#    fs4=df.rtn_ma66_pct<0.7
#    fs5=df.hv_m2y_pct>0.5
#    
#    fz1=df.hv_252_pct<0.3
##    fz2=df.hv_66_pct>0.4
#    fz3=df.hv_22_pct>0.4
##    fz4=df.hv_252<0.225
##    fz5=df.hv_66_pct<0.4
#    fz6=df.rtn_ma66_pct>0.7
#    
##    fz6=df.sharpe_66_pct>0.5
##    fz7=df.p_252>0.001
##    fz8=df.sharpe_5_pct>=0.5
#    fz9=df.hv_5m<0.83
#     
#    dl=df[(fl1 &  fl4 &fl2 )]
#    ds=df[(fs1 &   fs3 & fs4 & fs5)]
#    dz=df[(fz1 & fz9 & fz3 )]
##    dz=df[(fz6 & fz7 & fz8 & fz9 &fz1)]
    
#etf
#    el1=df.rtn_ma252_pct<0.3  #diff from L
#    el2=df.hv_252_pct>0.5
#    el3=df.hv_66_pct>0.6
#    el4=df.hv_22_pct>0.6
#    el5=df.hv_m2y_pct<0.5
#    
#    es1=df.rtn_ma252_pct>0.8  #diff from L
#    es2=df.hv_252_pct>0.8
#    es3=df.hv_66_pct>0.7
#    es4=df.hv_22_pct>0.7
#    es5=df.hv_m2y_pct>0.5
    
    ez1=df.rtn_ma66_pct>0.4
    ez2=df.rtn_ma66_pct<0.7
    ez3=df.hv_252_pct>0.8
    ez4=df.hv_66_pct<0.6
    ez5=df.hv_22_pct<0.6
    ez6=df.rtn_22_pct>0.3
    ez7=df.rtn_22_pct<0.75
    ez8=df.hv_5m<0.83
    ez9=df.hv_252<0.25
    
    
#    el=df[el1 & el2 & el3 & el4 & el5]
#    es=df[es1 & es2 & es3 & es4 & es5]
    ez=df[ez8  & ez3 &  ez1 & ez2]
    #ez=df[(ez3) & ez8]
#    dl=el
#    ds=es
    dz=ez
    
    tl=dl[dl.rtn_22_fwd_pct>=0.8].shape[0]
    ts=ds[ds.rtn_22_fwd_pct<=0.2].shape[0]
    tz=dz[np.abs(dz.rtn_22_fwd_pct)<=0.2].shape[0]
    vet_l=tl/dl.shape[0]
    vet_s=ts/ds.shape[0]
    vet_z=tz/dz.shape[0]
    
#    print("L- true_l: %s  sum_l: %s  vet_l:%s"%(tl, dl.shape[0], vet_l))
#    print("S- true_s: %s  sum_s: %s  vet_s:%s"%(ts, ds.shape[0], vet_s))
    print("Z- true_z: %s  sum_z: %s  vet_z:%s"%(tz, dz.shape[0], vet_z))

#df=data_import('sp500')
#do, df, dx, dl, ds, dz, dnl, dns, dnz, len=plot_prep(df)
#feature_vet(df)
#kde_plot(len)
#feature_vet(df)
df=pd.read_csv(r'G:\Trading\Trade_python\pycode\pytest\playlist_etf_test_2017-02-16_100.csv')
dc=df
dc['rv_5']=dc['rtn_5']/dc['hv_5']
dc['rv_22']=dc['rtn_22']/dc['hv_22']
dc['rv_66']=dc['rtn_66']/dc['hv_66']
dc['ztn_5_fwd']=dc['p_5_fwd']/dc['close_qdate']-1
dc['ztn_10_fwd']=dc['p_10_fwd']/dc['close_qdate']-1
dc['ztn_22_fwd']=dc['p_22_fwd']/dc['close_qdate']-1
df1=dc[['hv_5', 'hv_66','hv_22', 'hv_252','p_252','rtn_22', 'rtn_5', 'rtn_66',\
    'mean_510', 'mean_1022','mean_2266', 'mean_66252','p_5_fwd', 'p_10_fwd', 'p_22_fwd',\
        'rv_5', 'rv_22', 'rv_66', 'ztn_5_fwd','ztn_10_fwd', 'ztn_22_fwd']]
df1['momt']=df1.mean_510.astype(str)+df1.mean_1022.astype(str)+df1.mean_2266.astype(str)+df1.mean_66252.astype(str)
df1['momt_rank']=df1['momt'].rank(pct=True)*0.01
df1=df1.drop(['mean_510', 'mean_1022','mean_2266', 'mean_66252','momt'], axis=1)
df1=df1.sort_index(axis=1)
df1.dropna(inplace=True)

x_22=df1.iloc[:,:15]
y_22=df1['ztn_22_fwd']
x_22a=x_22.drop(['hv_22', 'hv_252', 'hv_5', 'hv_66','p_10_fwd', 'p_22_fwd','p_5_fwd', 'rtn_22', 'rtn_5', 'rtn_66'],\
    axis=1)

def ml_importance(x,y):
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    model_c=ExtraTreesClassifier()
    model_r=ExtraTreesRegressor()
    x=x.values
    y=y.values
    #model_c.fit(x,y)
    model_r.fit(x,y)
    print(model_r.feature_importances_)

def ml_pca(x,y):
    from sklearn.decomposition import PCA
    pca=PCA(n_components=5)
    fit=pca.fit(x,y)
# summarize component
    print("Explained Variance: %s"%fit.explained_variance_ratio_)
    print(fit.components_)      

def ml_rfe(x,y): #recursive feature elimination
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    model=LogisticRegression()
    rfe=RFE(model,3)
    x=x.values
    y=y.values
    fit=rfe.fit(x,y)
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)
    