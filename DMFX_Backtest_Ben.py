
# coding: utf-8

# In[1]:

import os
import sys
import time


# In[2]:

start_time = time.time()


# In[3]:

# add Eureka to path
if os.name == 'nt':
    sys.path.append(os.path.abspath('../../eureka'))
    
if os.name == 'posix':
    import mkl
    mkl.set_num_threads(mkl.get_max_threads())


# In[4]:

import pandas as pd
import numpy as np
import datetime as dt


# In[5]:

from eureka.risk import calc_ewma_riskmodel
from eureka.signal import calc_zscore, score_to_alpha
from eureka.optimize import mean_variance_optimizer
from eureka.backtest import backtest_metrics
from eureka.report import backtest_report, aggregate_report


# In[ ]:

# module to reload modules without having to restart the kernel
get_ipython().magic('load_ext autoreload')


# In[ ]:

get_ipython().magic('autoreload 2')


# In[ ]:

# to plot the graphs in the notebook itself
get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (12, 9)


# ### Configuration

# In[ ]:

cparams = {}

# strategy
cparams['strategy'] = 'DMFX'

# asset list
cparams['assets'] = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NOK', 'NZD', 'SEK']
cparams['assets_with_USD'] = ['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NOK', 'NZD', 'SEK', 'USD']
cparams['forward_tenor'] = '3M'
    
# date parameters
cparams['run_dt'] =  dt.datetime(2016, 5, 31) #dt.date.today()
cparams['start_dt'] = dt.datetime(1999, 1, 1)
cparams['end_dt'] = (cparams['run_dt'] - pd.tseries.offsets.BDay(1)).to_datetime()
cparams['freq'] = 'B' # business-days
cparams['dt_range'] = pd.date_range(start=cparams['start_dt'],end=cparams['end_dt'],freq=cparams['freq'],normalize=True)


# In[ ]:

# risk model parameters 
rmparams = {}
rmparams['vol_halflife'] = 21
rmparams['corr_halflife'] = 126
rmparams['min_periods'] = 252
rmparams['corr_cap'] = 1.0
rmparams['corr_type'] = 'shrinktoaverage'
rmparams['lag'] = 0
cparams['riskmodel_params'] = rmparams


# In[ ]:

# optimization parameters
optoparams = {}
optoparams['riskmodel'] = 'SHRINKAVG'
optoparams['risk_aversion'] = 200
optoparams['risk_aversion_tcost'] = 85
optoparams['tcost_aversion'] = 0.05
optoparams['position_lower_bound'] = -1.0
optoparams['position_upper_bound'] = 1.0
optoparams['portfolio_lower_bound'] = -0.25
optoparams['portfolio_upper_bound'] = 0.25
cparams['opto_params'] = optoparams


# In[ ]:

# insight names
cparams['core_insights'] = ['LII']
cparams['insights'] = cparams['core_insights']

insight_lag = 2

cparams['insight_params'] = {}
cparams['insight_params']['IC'] = 0.1
for insight in cparams['insights']:
    iparams = {}
    if insight == 'LII':
        iparams['smth_halflife'] = 3
        iparams['zsc_halflife'] = 5
        iparams['zsc_seedperiod'] = 5
        iparams['subtract_mean'] = True
        iparams['score_cap'] = 3.0
        iparams['score_lag'] = insight_lag
        iparams['sign'] = -1.0
    cparams['insight_params'][insight] = iparams


# ### Generate Returns

# In[ ]:

# load returns
#ret_df = pd.read_csv(...)


# ### Generate Risk Model

# In[ ]:

riskmodel_dict = calc_ewma_riskmodel(ret_df,
                                          vol_halflife=cparams['riskmodel_params']['vol_halflife'],
                                          vol_seed_period=cparams['riskmodel_params']['vol_halflife'],
                                          corr_halflife=cparams['riskmodel_params']['corr_halflife'],
                                          corr_seed_period=cparams['riskmodel_params']['corr_halflife'],
                                          corr_type=cparams['riskmodel_params']['corr_type'],
                                          corr_cap=cparams['riskmodel_params']['corr_cap'],
                                          lag=cparams['riskmodel_params']['lag'])


# ### Generate Insights

# In[ ]:

insightdict = {}
scoredict = {}
alphadict = {}

for insight in cparams['core_insights']:
    print('Generating insight : ',insight)
    if insight == 'LII':
        insightdict[insight] = None #function to compute insight
        scoredict[insight] = calc_zscore(insightdict[insight], 
                                    mean_halflife=cparams['insight_params'][insight]['zsc_halflife'], 
                                    std_halflife=cparams['insight_params'][insight]['zsc_halflife'], 
                                    mean_seed_period=cparams['insight_params'][insight]['zsc_seedperiod'], 
                                    std_seed_period=cparams['insight_params'][insight]['zsc_seedperiod'], 
                                    smth_halflife=cparams['insight_params'][insight]['smth_halflife'], 
                                    subtract_mean=cparams['insight_params'][insight]['subtract_mean'],
                                    cap=cparams['insight_params'][insight]['score_cap'], 
                                    lag=cparams['insight_params'][insight]['score_lag'])      
     
    # apply score sign
    scoredict[insight] = scoredict[insight] * cparams['insight_params'][insight]['sign']
    # score to alpha
    alphadict[insight] = score_to_alpha(scoredict[insight], 
                                        riskmodel_dict['vol'], 
                                        IC = cparams['insight_params']['IC'])


# ### Generate Holdings

# In[ ]:

holdingdict = {}

for insight in cparams['insights']:   
    print('Generating holdings for insight : ',insight)
    holdingdict[insight] = mean_variance_optimizer(alphadict[insight], 
                                                   riskmodel_dict['cov'], 
                                                   risk_aversion = cparams['opto_params']['risk_aversion'], 
                                                   position_lower_bound = cparams['opto_params']['position_lower_bound'],
                                                   position_upper_bound = cparams['opto_params']['position_upper_bound'])


# ### Backtest Metrics

# In[ ]:

backtestdict = {}
    
for insight in cparams['insights']:
    print('Computing backtest metrics for insight : ', insight)
    backtestdict[insight] = backtest_metrics(ret_df,
                                                 holdingdict[insight],
                                                 riskmodel_dict,
                                                 scoredict[insight],
                                                 alphadict[insight],                                                 
                                                 risk_aversion=cparams['opto_params']['risk_aversion'])


# ### Generate Aggregate Report

# In[ ]:

for insight in cparams['insights']:
    print('Generating aggregate report for insight : ', insight)
    aggregate_report(backtestdict[insight],
                     title=insight,
                     author='AEG',
                     filename=insight+'_'+cparams['returns_source']+'.pdf')


# In[ ]:




# In[ ]:



