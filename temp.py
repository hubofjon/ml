# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:42:26 2017

@author: liq6
"""

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
#df=pd.read_csv(r'P:\Documents\Python Scripts\playlist_etf_test_2017-02-16.csv')
df=pd.read_csv(r'G:\Trading\Trade_python\pycode\pytest\playlist_etf_test_2017-02-15.csv')

#dc=dc.drop(['p_5_fwd','p_10_fwd'], axis=1)
df['rv_5']=df['rtn_5']/df['hv_5']
df['rv_22']=df['rtn_22']/df['hv_22']
df['rv_66']=df['rtn_66']/df['hv_66']
df['ztn_22_fwd']=df['p_22_fwd']/df['close_qdate']-1


#cateogrize

con_L=(df['p_22_fwd']-df['close_qdate'])/df['p_22_sig']>=1.2
con_S=(df['p_22_fwd']-df['close_qdate'])/df['p_22_sig']<= -1.2
con_Z=np.abs(df['ztn_22_fwd'])<=0.02

df.loc[con_L, 'trend']='L2'
df.loc[con_S, 'trend']='S2'
df.loc[con_Z, 'trend']='Z2'

dc=df
df1=df[['hv_5', 'hv_66','hv_22', 'hv_252','p_252','rtn_22', 'rtn_5', 'rtn_66',\
    'mean_510', 'mean_1022','mean_2266', 'mean_66252',\
        'rv_5', 'rv_22', 'rv_66',  'ztn_22_fwd','trend']]
df1['momt']=df1.mean_510.astype(str)+df1.mean_1022.astype(str)+df1.mean_2266.astype(str)+df1.mean_66252.astype(str)
df1['momt_rank']=df1['momt'].rank(pct=True)*0.01
df1=df1.drop(['mean_510', 'mean_1022','mean_2266', 'mean_66252','momt'], axis=1)
df1=df1.sort_index(axis=1)
df1.dropna(inplace=True)
#df1=df1[df1.p_252<=0.05]
x=df1.iloc[:,:12]
yr=df1['ztn_22_fwd']
yc=df1['trend']

x0=x.drop(['rv_22','rv_5','rv_66'], axis=1)
x1=x[['hv_22', 'hv_252', 'hv_5', 'hv_66', 'momt_rank', 'p_252', 'rv_22',\
       'rv_5', 'rv_66']]
x2=x[['p_252','hv_66','hv_252','rv_5','rv_22', 'rv_66']]
x3=x2.drop(['rv_22', 'rv_5'], axis=1)
x4=x[['p_252','hv_66','hv_252','rtn_22', 'rtn_66']]
def ml_fi(x,y,option='r'):
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    if option =='c': #regression
        model=ExtraTreesClassifier()
    elif option =='r': #classifer
        model=ExtraTreesRegressor()
    x=x.values
    y=y.values
    x_train, x_test, y_train, y_test= train_test_split(x,y, train_size = 0.8)
    #model_c.fit(x,y)
    model_fi=model.fit(x_train,y_train)
    pickle.dump(model_fi, open(r'G:\Trading\Trade_python\pycode\pytest\model_fi_etf.dat',"wb"))
    y_test_pred=model_fi.predict(x_test)
    print(model.feature_importances_)
    return y_test_pred, y_test

def ml_apply_fi():#df is df_stat (etf)
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    import pickle
    
    model_fi=pickle.load(open(r'G:\Trading\Trade_python\pycode\pytest\model_fi_etf.dat',"rb"))
    qry="SELECT * FROM tbl_stat_etf"
    df=read_sql(qry, lday)
    x=df[['p_252','hv_66','hv_252','rtn_22', 'rtn_66']].values
    y=model_fi.predict(x)
    y=y_pred[:,None]  #convert to array[n,1]
    y=pd.DataFrame(y,columns=['play'])
    # dc=pd.DataFrame([y_pred, y], columns=['y_pred','y'])
    #df_play_ml=np.concatenate((x, y_pred), axis=0)   
    df_play_ml=pd.concat([df,y], axis=1)
    return df_play_ml
    
    
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
    
def ml_line(x,y):
    lr = linear_model.LinearRegression()
       #y=y.applymap(lambda x:1 if x else 0)
    x=x.values
#    x=pre_process(x)
    y=y.values

    x_train, x_test, y_train, y_test= train_test_split(x,y, train_size = 0.8)
    #print (x_train.shape, y_train.shape)
#standardize features
#    scaler= preprocessing.StandardScaler().fit(x_train)    
#    x_train=scaler.transform(x_train)
#    x_test=scaler.transform(x_test)
    
    model_lr=lr.fit(x_train, y_train)
    pickle.dump(model_lr, open(r'G:\Trading\Trade_python\pycode\pytest\model_lr_etf.dat',"wb"))
    y_test_pred=lr.predict(x_test)
    error=y_test-y_test_pred
    
    import sklearn.metrics as sm
    print ("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2) )
    return y_test_pred, y_test, error