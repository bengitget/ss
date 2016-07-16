
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 09:20:44 2016

@author: e612727
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import datetime as dt
import xlwt



def inflaZ(data, sub_date=0, transformation=0):
    c = [x for x in data.iloc[:,0] if str(x) != 'nan']    
    n=len(c)
    data=data.iloc[0:n,0:4]
    b= data.iloc[:,2] 
    c = [x for x in data.iloc[:,0] if str(x) != 'nan']           
          
    m=len(b)
    x=[]
    h=b.fillna('na')
    for i in range(1,m):
        a=data.iloc[i-1,0]
        if  h[i]=='na':
            i=+1
        else:
            h[i]=int(h[i])
            c=str(h[i])
            d=dt.datetime.strptime(c,'%Y%m%d') 
            t=d-a
            x.append(t)
                
    if x==[]:
        delay=dt.timedelta(15)
    else:
        delay=np.mean(x)   
 

    
    h=b.fillna('na')
    new_date1=[]
    for i in range(1,n+1):
        if h[i]=='na':
            new_date=data.iloc[i-1,0]+delay
            new_date=new_date.replace(minute=0, hour=0, second=0, microsecond=0)
            i=+1
            new_date1.append(new_date)
     
        else:
            h[i]=int(h[i])
            c=str(h[i])
            d=dt.datetime.strptime(c,'%Y%m%d') 
            new_date1.append(d)
    
      
    if sub_date!=[]:
        if x==[]: 
             new_date1=[]
             new_date1=sub_date
#            
    else:
        new_date1=new_date1

    p=data.iloc[:,3]
    p=p.fillna('na')
    new_release1=[]
    for i in range(0,n):
        if p[i+1]=='na':
            new_release=data.iloc[i,1]
            i=+1
            new_release1.append(new_release)
     
        else:
         new_release1.append(data.iloc[i,3])
   
  
    if transformation=="Diff1M":
        new_release1=pd.DataFrame(new_release1).diff() #to calculate 1M difference
    else:
        new_release1=new_release1
         
    smth_halflife=3
    zsc_halflife=6
    subtract_mean=True
    score_cap=3
    score_lag=0
    zscore=[]
    cparams = {}
    k=pd.DataFrame(new_release1)
    k.index=new_date1
   
        
    #new_release1=pd.DataFrame(new_release1)
    #new_release1.index=new_date1
   
    data['Updated Date']=new_date1
    data['Updated Release']=new_release1
    data.columns=['PCE Date','Price','Eco.Release.Date','Actual Release','Update Release Date','Update Release']                        
                            
                           
                            

    return k, data
    
    
    
    
def usalphaIC (release,
               asset,
               fields,
#               riskmodel,
#               index,
               col=0,
               sign=0):

    smth_halflife=3
    zsc_halflife=6
    subtract_mean=True
    score_cap=3
    score_lag=0
    zscore=[]
    cparams = {}
    c = [x for x in release.iloc[:,0] if str(x) != 'nan']    
    n=len(c)
    release=release.iloc[0:n,0:6]
#    riskmodel=riskmodel_dict
    k=release
    fields=fields
    if col==1:
        k=k
       
    else:
       
        i=len(k.columns)
        k=pd.DataFrame(release.iloc[:,(i-1)])
        k.index=release.iloc[:,(i-2)]
        
  
   
   
    
                                    
    if len(fields)==1:
        us2y=pd.DataFrame(asset[fields[0]])
        final_z.columns=[fields[0]]
        l=pd.DataFrame(riskmodel_dict['vol'][fields[0]])
        alpha_2 = score_to_alpha(final_z, l, IC = 0.1)
    
        IC_2, pval_2 = calc_realized_IC(us2y, 
                                      alpha_2, 
                                      l)
                                
        alpha_2=alpha_2
        IC=[IC_2]
        pval=[pval_2]
        
    elif len(fields)==2:
        us2y=pd.DataFrame(asset[fields[0]])
        final_z.columns=[fields[0]]
        l=pd.DataFrame(riskmodel_dict['vol'][fields[0]])
        alpha_2 = score_to_alpha(final_z, l, IC = 0.1)
        IC_2, pval_2 = calc_realized_IC(us2y, 
                                      alpha_2, 
                                      l)
        us5y=pd.DataFrame(asset[fields[1]])
        final_z.columns=[fields[1]]
        l=pd.DataFrame(riskmodel_dict['vol'][fields[1]])
        
        alpha_5 = score_to_alpha(final_z, l, IC = 0.1)
    
        IC_5, pval_5 = calc_realized_IC(us5y, 
                                      alpha_5, 
                                      l)
        alpha_2['alpha 2']=alpha_5
        IC=[IC_2,IC_5]
        pval=[pval_2,pval_5]
        
    elif len(fields)==3:
        us2y=pd.DataFrame(asset[fields[0]])
        final_z.columns=[fields[0]]
        l=pd.DataFrame(riskmodel_dict['vol'][fields[0]])
        alpha_2 = score_to_alpha(final_z, l, IC = 0.1)
        IC_2, pval_2 = calc_realized_IC(us2y, 
                                      alpha_2, 
                                      l)
        us5y=pd.DataFrame(asset[fields[1]])
        final_z.columns=[fields[1]]
        l=pd.DataFrame(riskmodel_dict['vol'][fields[1]])
        alpha_5 = score_to_alpha(final_z, l, IC = 0.1)
        IC_5, pval_5 = calc_realized_IC(us5y, 
                                      alpha_5, 
                                      l)
        us10y=pd.DataFrame(asset[fields[2]])
        final_z.columns=[fields[2]]
        l=pd.DataFrame(riskmodel_dict['vol'][fields[2]])
        
        alpha_10 = score_to_alpha(final_z, l, IC = 0.1)
    
        IC_10, pval_10 = calc_realized_IC(us10y, 
                                      alpha_10, 
                                      l)
        alpha_2['alpha 2']=alpha_5        
        alpha_2['alpha 3']=alpha_10
        IC=[IC_2,IC_5,IC_10]
        pval=[pval_2,pval_5,pval_10]
    elif len(fields)==4:
        us2y=pd.DataFrame(asset[fields[0]])
        final_z.columns=[fields[0]]
        l=pd.DataFrame(riskmodel_dict['vol'][fields[0]])
        alpha_2 = score_to_alpha(final_z, l, IC = 0.1)
        IC_2, pval_2 = calc_realized_IC(us2y, 
                                      alpha_2, 
                                      l)
        us5y=pd.DataFrame(asset[fields[1]])
        final_z.columns=[fields[1]]
        l=pd.DataFrame(riskmodel_dict['vol'][fields[1]])
        alpha_5 = score_to_alpha(final_z, l, IC = 0.1)
        IC_5, pval_5 = calc_realized_IC(us5y, 
                                      alpha_5, 
                                      l)
        us10y=pd.DataFrame(asset[fields[2]])
        final_z.columns=[fields[2]]
        l=pd.DataFrame(riskmodel_dict['vol'][fields[2]])
        
        alpha_10 = score_to_alpha(final_z, l, IC = 0.1)
    
        IC_10, pval_10 = calc_realized_IC(us10y, 
                                      alpha_10, 
                                      l)
        
        us20y=pd.DataFrame(asset[fields[3]])
        final_z.columns=[fields[3]]
        l=pd.DataFrame(riskmodel_dict['vol'][fields[3]])
        
    
        alpha_20 = score_to_alpha(final_z, l, IC = 0.1)
    
        IC_20, pval_20 = calc_realized_IC(us20y, 
                                      alpha_20, 
                                      l)
        alpha_2['alpha 2']=alpha_5        
        alpha_2['alpha 3']=alpha_10
        alpha_2['alpha 4']=alpha_20
        IC=[IC_2,IC_5,IC_10,IC_20]
        pval=[pval_2,pval_5,pval_10,pval_20]
    elif len(fields)==5:
        us2y=pd.DataFrame(asset[fields[0]])
        final_z.columns=[fields[0]]
        l=pd.DataFrame(riskmodel_dict['vol'][fields[0]])
        alpha_2 = score_to_alpha(final_z, l, IC = 0.1)
        IC_2, pval_2 = calc_realized_IC(us2y, 
                                      alpha_2, 
                                      l)
        us5y=pd.DataFrame(asset[fields[1]])
        final_z.columns=[fields[1]]
        l=pd.DataFrame(riskmodel_dict['vol'][fields[1]])
        alpha_5 = score_to_alpha(final_z, l, IC = 0.1)
        IC_5, pval_5 = calc_realized_IC(us5y, 
                                      alpha_5, 
                                      l)
        us10y=pd.DataFrame(asset[fields[2]])
        final_z.columns=[fields[2]]
        l=pd.DataFrame(riskmodel_dict['vol'][fields[2]])
        
        alpha_10 = score_to_alpha(final_z, l, IC = 0.1)
    
        IC_10, pval_10 = calc_realized_IC(us10y, 
                                      alpha_10, 
                                      l)
        
        us20y=pd.DataFrame(asset[fields[3]])
        final_z.columns=[fields[3]]
        l=pd.DataFrame(riskmodel_dict['vol'][fields[3]])
        
    
        alpha_20 = score_to_alpha(final_z, l, IC = 0.1)
    
        IC_20, pval_20 = calc_realized_IC(us20y, 
                                      alpha_20, 
                                      l)
        us30y=pd.DataFrame(asset[fields[4]])
        final_z.columns=[fields[4]]
        l=pd.DataFrame(riskmodel_dict['vol'][fields[4]])
        
        alpha_30 = score_to_alpha(final_z, l, IC = 0.1)
    
        IC_30, pval_30 = calc_realized_IC(us30y, 
                                      alpha_30, 
                                      l)
        alpha_2['alpha 2']=alpha_5        
        alpha_2['alpha 3']=alpha_10
        alpha_2['alpha 4']=alpha_20
        alpha_2['alpha 5']=alpha_30        
        IC=[IC_2,IC_5,IC_10,IC_20,IC_30] 
        pval=[pval_2,pval_5,pval_10,pval_20,pval_30]
    else:
        return "error"
    
    m=len(final_z)
    z_1=final_z[0:int(m/2)]
    z_2=final_z[int(m/2):]
    IC1=alphaIC(z_1,asset,fields)
    IC2=alphaIC(z_2,asset,fields)
    


    
    alpha_dict={}
    alpha_dict['alpha']=alpha_2
    alpha_dict['IC']=IC
    alpha_dict['pval']=pval
    alpha_dict['Z-score']=final_z
    alpha_dict['IC1']=IC1['IC']
    alpha_dict['pval1']=IC1['pval']
    alpha_dict['IC2']=IC2['IC']
    alpha_dict['pval2']=IC2['pval']
    
    
    return alpha_dict      
    
def compiledIC(alpha_dict):
    
    IC=pd.DataFrame(alpha_dict['IC'])
    IC['pval']=alpha_dict['pval']
    
    IC['IC1']=alpha_dict['IC1']
    IC['pval1']=alpha_dict['pval1']
    IC['IC2']=alpha_dict['IC2']
    IC['pval2']=alpha_dict['pval2']
    
    return IC
    
def alphaIC (release,
               ust,
               fields,
               transformation=0):

    smth_halflife=3
    zsc_halflife=6
    subtract_mean=True
    score_cap=3
    score_lag=0
    zscore=[]
    cparams = {}
#
#
      
    final_z=release
    

    ust=ust.reindex(index=final_z.index)
    vol=riskmodel_dict['vol'].reindex(index=final_z.index)  
                              
    if len(fields)==1:
        us2y=pd.DataFrame(ust[fields[0]])
        final_z.columns=[fields[0]]
        l=pd.DataFrame(vol[fields[0]])
        alpha_2 = score_to_alpha(final_z, l, IC = 0.1)
    
        IC_2, pval_2 = calc_realized_IC(us2y, 
                                      alpha_2, 
                                      l)
                                
        alpha_2=alpha_2
        IC=IC_2
        pval=pval_2
        
    elif len(fields)==2:
        us2y=pd.DataFrame(ust[fields[0]])
        final_z.columns=[fields[0]]
        l=pd.DataFrame(vol[fields[0]])
        alpha_2 = score_to_alpha(final_z, l, IC = 0.1)
        IC_2, pval_2 = calc_realized_IC(us2y, 
                                      alpha_2, 
                                      l)
        us5y=pd.DataFrame(ust[fields[1]])
        final_z.columns=[fields[1]]
        l=pd.DataFrame(vol[fields[1]])
        
        alpha_5 = score_to_alpha(final_z, l, IC = 0.1)
    
        IC_5, pval_5 = calc_realized_IC(us5y, 
                                      alpha_5, 
                                      l)
        alpha_2['alpha 2']=alpha_5
        IC=[IC_2,IC_5]
        pval=[pval_2,pval_5]
        
    elif len(fields)==3:
        us2y=pd.DataFrame(ust[fields[0]])
        final_z.columns=[fields[0]]
        l=pd.DataFrame(vol[fields[0]])
        alpha_2 = score_to_alpha(final_z, l, IC = 0.1)
        IC_2, pval_2 = calc_realized_IC(us2y, 
                                      alpha_2, 
                                      l)
        us5y=pd.DataFrame(ust[fields[1]])
        final_z.columns=[fields[1]]
        l=pd.DataFrame(vol[fields[1]])
        
        alpha_5 = score_to_alpha(final_z, l, IC = 0.1)
    
        IC_5, pval_5 = calc_realized_IC(us5y, 
                                      alpha_5, 
                                      l)
        us10y=pd.DataFrame(ust[fields[2]])
        final_z.columns=[fields[2]]
        l=pd.DataFrame(vol[fields[2]])
        alpha_10 = score_to_alpha(final_z, l, IC = 0.1)
    
        IC_10, pval_10 = calc_realized_IC(us10y, 
                                      alpha_10, 
                                      l)
        alpha_2['alpha 2']=alpha_5        
        alpha_2['alpha 3']=alpha_10
        IC=[IC_2,IC_5,IC_10]
        pval=[pval_2,pval_5,pval_10]
    elif len(fields)==4:
        us2y=pd.DataFrame(ust[fields[0]])
        final_z.columns=[fields[0]]
        l=pd.DataFrame(vol[fields[0]])
        alpha_2 = score_to_alpha(final_z, l, IC = 0.1)
        IC_2, pval_2 = calc_realized_IC(us2y, 
                                      alpha_2, 
                                      l)
        us5y=pd.DataFrame(ust[fields[1]])
        final_z.columns=[fields[1]]
        l=pd.DataFrame(vol[fields[1]])
        alpha_5 = score_to_alpha(final_z, l, IC = 0.1)
        IC_5, pval_5 = calc_realized_IC(us5y, 
                                      alpha_5, 
                                      l)
        us10y=pd.DataFrame(ust[fields[2]])
        final_z.columns=[fields[2]]
        l=pd.DataFrame(vol[fields[2]])
        
        alpha_10 = score_to_alpha(final_z, l, IC = 0.1)
    
        IC_10, pval_10 = calc_realized_IC(us10y, 
                                      alpha_10, 
                                      l)
        
        us20y=pd.DataFrame(ust[fields[3]])
        final_z.columns=[fields[3]]
        l=pd.DataFrame(vol[fields[3]])
        
    
        alpha_20 = score_to_alpha(final_z, l, IC = 0.1)
    
        IC_20, pval_20 = calc_realized_IC(us20y, 
                                      alpha_20, 
                                      l)
        alpha_2['alpha 2']=alpha_5        
        alpha_2['alpha 3']=alpha_10
        alpha_2['alpha 4']=alpha_20
        IC=[IC_2,IC_5,IC_10,IC_20]
        pval=[pval_2,pval_5,pval_10,pval_20]
    elif len(fields)==5:
        us2y=pd.DataFrame(ust[fields[0]])
        final_z.columns=[fields[0]]
        l=pd.DataFrame(vol[fields[0]])
        alpha_2 = score_to_alpha(final_z, l, IC = 0.1)
        IC_2, pval_2 = calc_realized_IC(us2y, 
                                      alpha_2, 
                                      l)
        us5y=pd.DataFrame(ust[fields[1]])
        final_z.columns=[fields[1]]
        l=pd.DataFrame(vol[fields[1]])
        alpha_5 = score_to_alpha(final_z, l, IC = 0.1)
        IC_5, pval_5 = calc_realized_IC(us5y, 
                                      alpha_5, 
                                      l)
        us10y=pd.DataFrame(ust[fields[2]])
        final_z.columns=[fields[2]]
        l=pd.DataFrame(vol[fields[2]])
        
        alpha_10 = score_to_alpha(final_z, l, IC = 0.1)
    
        IC_10, pval_10 = calc_realized_IC(us10y, 
                                      alpha_10, 
                                      l)
        
        us20y=pd.DataFrame(ust[fields[3]])
        final_z.columns=[fields[3]]
        l=pd.DataFrame(vol[fields[3]])
        
    
        alpha_20 = score_to_alpha(final_z, l, IC = 0.1)
    
        IC_20, pval_20 = calc_realized_IC(us20y, 
                                      alpha_20, 
                                      l)
        us30y=pd.DataFrame(ust[fields[4]])
        final_z.columns=[fields[4]]
        l=pd.DataFrame(vol[fields[4]])
        
        alpha_30 = score_to_alpha(final_z, l, IC = 0.1)
    
        IC_30, pval_30 = calc_realized_IC(us30y, 
                                      alpha_30, 
                                      l)
        alpha_2['alpha 2']=alpha_5        
        alpha_2['alpha 3']=alpha_10
        alpha_2['alpha 4']=alpha_20
        alpha_2['alpha 5']=alpha_30        
        IC=[IC_2,IC_5,IC_10,IC_20,IC_30] 
        pval=[pval_2,pval_5,pval_10,pval_20,pval_30]
    else:
        return "error"
    
     
#    alpha_dict={}
#    alpha_dict['alpha']=alpha_2
#    alpha_dict['IC']=IC
#    alpha_dict['pval']=pval
#    alpha_dict['Z-score']=final_z
    return IC, pval     