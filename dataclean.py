# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 11:06:43 2016

@author: e612727
"""

import pandas as pd
import numpy
from datetime import datetime, timedelta
import datetime as dt
import xlwt

data=pd.ExcelFile('growth indicators.xlsx',sheet_name="sheet2",na_values=['NA'])
dmfx=pd.read_csv('dmfx_returns.csv')
rates_return=pd.read_csv('rates_futures_returns.csv',na_values=['NA'])

#rates_return=pd.read_csv('rates_futures_returns.csv',na_values=['NA'])
us=data.parse(0)
eu=data.parse(1)
japan=data.parse(2) # up to 33
nz=data.parse(3)
nw=data.parse(4)
sw=data.parse(5)
sz=data.parse(6)
uk=data.parse(7) # up to 33
au=data.parse(8)
canada=data.parse(9)


start_dt = dt.datetime(1999, 1, 1)
end_dt =  dt.datetime(2016, 4, 29)
freq = 'B' # business-days

dt_range = pd.date_range(start=start_dt, end=dt.datetime(2015,12,17), freq=freq, normalize=True)
fields=['AUD']
fx=dmfx['AUD']
fx=pd.DataFrame(fx)
fx.index=dt_range

#dt_range = pd.date_range(start=start_dt, end=end_dt, freq=freq, normalize=True)
#fields=['JPY_10Y']
#ust=rates_return['JPY_10Y']
#ust=pd.DataFrame(ust)
#ust.index=dt_range

rmparams = {}
rmparams['vol_halflife'] = 21
rmparams['corr_halflife'] = 126
rmparams['corr_cap'] = 1.0
rmparams['corr_type'] = 'shrinktoaverage'
rmparams['lag'] = 0


riskmodel_dict = calc_ewma_riskmodel(fx,
                          vol_halflife=rmparams['vol_halflife'],
                          vol_seed_period=rmparams['vol_halflife'],
                           corr_halflife=rmparams['corr_halflife'],
                            corr_seed_period=rmparams['corr_halflife'],
                            corr_type=rmparams['corr_type'],
                            corr_cap=rmparams['corr_cap'],
                            lag=rmparams['lag'])

x=[]
ic=[]
pval=[]
n=int((len(au.columns)+1)/5)
fx=pd.DataFrame(fx)
l=pd.DataFrame(riskmodel_dict['vol'][fields[0]])
for i in range(0,n):

    input=au.iloc[1:,5*i:5*i+4]
    
    update,full=inflaZ(input,transformation=1)
    
    smth_halflife=3
    zsc_halflife=6
    subtract_mean=True
    score_cap=3
    score_lag=0
    zscore=[]
    cparams = {}
    
    
         
    score_df=calc_zscore(update,
                         mean_halflife=zsc_halflife, 
                         std_halflife=zsc_halflife, 
                         mean_seed_period=zsc_halflife,
                         std_seed_period=zsc_halflife,
                         smth_halflife=smth_halflife,
                         subtract_mean=subtract_mean,
                         cap=score_cap, 
                         lag=score_lag)
  
    
    score_df1 = score_df.reindex(index=dt_range)
    score_df1 = score_df1.ffill(axis=0)   
    final_z = score_df1.shift(2)
    final_z=final_z*(-1)
    
   
    final_z.columns=[fields[0]]
       
    alpha = score_to_alpha(final_z, l, IC = 0.1)
    IC1, pval1 = calc_realized_IC(fx, alpha, l)
                                
    m=len(final_z)
    z_1=final_z[0:int(m/2)]
    z_2=final_z[int(m/2):]
    IC_1,pval_1=alphaIC(z_1,fx,fields)
    IC_2,pval_2=alphaIC(z_2,fx,fields)
#    alpha1=score_to_alpha(z_1, l, IC = 0.1)
#    IC_1, pval_1 = calc_realized_IC(fx, alpha1, l)
#    
#    alpha2 = score_to_alpha(z_2, l, IC = 0.1)
#    IC_2, pval_2 = calc_realized_IC(fx, alpha2, l)
#   
                                
    ic.append(IC1)
    ic.append(pval1)
    ic.append(IC_1)
    ic.append(pval_1)
    ic.append(IC_2)
    ic.append(pval_2)

ic=pd.DataFrame(ic)
ic.to_csv('temp result.csv')    
   
m=pd.concat(x,axis=1)
   
mm=m.mean(axis=1)    
    