
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
            c=str(h[i])
            d=dt.datetime.strptime(c,'%Y%m%d') 
            t=d-a
            x.append(t)
    if x==[]:
        delay=dt.timedelta(0)
    else:
        delay=np.mean(x)   
    #np.take(0,np.where(np.isnan(delay)))
   
        
    
    h=b.fillna('na')
    new_date1=[]
    for i in range(1,n+1):
        if h[i]=='na':
            new_date=data.iloc[i-1,0]+delay
            new_date=new_date.replace(minute=0, hour=0, second=0, microsecond=0)
            i=+1
            new_date1.append(new_date)
     
        else:
            c=str(h[i])
            d=dt.datetime.strptime(c,'%Y%m%d') 
            new_date1.append(d)
            
    if x==[]:
        
        new_date1=[]
        new_date1=sub_date
        
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
   
  
    if transformation==1:
        
        new_release1=pd.DataFrame(new_release1).diff() #to calculate 1M difference
    else:
        new_release1=new_release1
         
#       
    #new_release1=pd.DataFrame(new_release1)
    #new_release1.index=new_date1
   
    data['Updated Date']=new_date1
    data['Updated Release']=new_release1
    data.columns=['PCE Date','Price','Eco.Release.Date','Actual Release','Update Release Date','Update Release']                        
                            
                            
                            

    return data