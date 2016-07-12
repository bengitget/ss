# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 11:36:36 2016

@author: e612727
"""

import pandas as pd
import numpy
from datetime import datetime, timedelta
import datetime as dt
import xlwt


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
        IC=[IC_2]
        pval=[pval_2]
        
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
        l=pd.DataFrame(riskmodel_dict['vol'][fields[0]])
        alpha_2 = score_to_alpha(final_z, l, IC = 0.1)
        IC_2, pval_2 = calc_realized_IC(us2y, 
                                      alpha_2, 
                                      l)
        us5y=pd.DataFrame(ust[fields[1]])
        final_z.columns=[fields[1]]
        l=pd.DataFrame(riskmodel_dict['vol'][fields[1]])
        alpha_5 = score_to_alpha(final_z, l, IC = 0.1)
        IC_5, pval_5 = calc_realized_IC(us5y, 
                                      alpha_5, 
                                      l)
        us10y=pd.DataFrame(ust[fields[2]])
        final_z.columns=[fields[2]]
        l=pd.DataFrame(riskmodel_dict['vol'][fields[2]])
        
        alpha_10 = score_to_alpha(final_z, l, IC = 0.1)
    
        IC_10, pval_10 = calc_realized_IC(us10y, 
                                      alpha_10, 
                                      l)
        
        us20y=pd.DataFrame(ust[fields[3]])
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
        us2y=pd.DataFrame(ust[fields[0]])
        final_z.columns=[fields[0]]
        l=pd.DataFrame(riskmodel_dict['vol'][fields[0]])
        alpha_2 = score_to_alpha(final_z, l, IC = 0.1)
        IC_2, pval_2 = calc_realized_IC(us2y, 
                                      alpha_2, 
                                      l)
        us5y=pd.DataFrame(ust[fields[1]])
        final_z.columns=[fields[1]]
        l=pd.DataFrame(riskmodel_dict['vol'][fields[1]])
        alpha_5 = score_to_alpha(final_z, l, IC = 0.1)
        IC_5, pval_5 = calc_realized_IC(us5y, 
                                      alpha_5, 
                                      l)
        us10y=pd.DataFrame(ust[fields[2]])
        final_z.columns=[fields[2]]
        l=pd.DataFrame(riskmodel_dict['vol'][fields[2]])
        
        alpha_10 = score_to_alpha(final_z, l, IC = 0.1)
    
        IC_10, pval_10 = calc_realized_IC(us10y, 
                                      alpha_10, 
                                      l)
        
        us20y=pd.DataFrame(ust[fields[3]])
        final_z.columns=[fields[3]]
        l=pd.DataFrame(riskmodel_dict['vol'][fields[3]])
        
    
        alpha_20 = score_to_alpha(final_z, l, IC = 0.1)
    
        IC_20, pval_20 = calc_realized_IC(us20y, 
                                      alpha_20, 
                                      l)
        us30y=pd.DataFrame(ust[fields[4]])
        final_z.columns=['fields[4]']
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
    
    
    alpha_dict={}
    alpha_dict['alpha']=alpha_2
    alpha_dict['IC']=IC
    alpha_dict['pval']=pval
    alpha_dict['Z-score']=final_z
    return alpha_dict      
    