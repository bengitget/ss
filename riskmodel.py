# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 17:27:16 2016

@author: e612727
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

def _calc_ewma_variance(ret_df,
                        vol_halflife = 252,
                        vol_seed_period = 252,
                        var_annualization_factor = 1,
                        lag = 1):
    '''
    Calculate EWMA (exponentially weighted moving average) variance
    '''
    if lag > 0:
        ret_df = ret_df.shift(lag)
    return pd.ewmvar(ret_df, halflife=vol_halflife, min_periods=vol_seed_period) * var_annualization_factor

def _calc_ewma_volatility(ret_df,
                          vol_halflife = 252,
                          vol_seed_period = 252,
                          var_annualization_factor = 1,
                          lag = 1):
    '''
    Calculate EWMA (exponentially weighted moving average) volatility
    '''
    if lag > 0:
        ret_df = ret_df.shift(lag)
    return pd.ewmvol(ret_df, halflife=vol_halflife, min_periods=vol_seed_period) * np.sqrt(var_annualization_factor)

def _calc_ewma_correlation(ret_df,
                           corr_halflife = 252,
                           corr_seed_period = 252,
                           lag = 1):
    '''
    Calculate EWMA (exponentially weighted moving average) correlation matrix
    '''
    if lag > 0:
        ret_df = ret_df.shift(lag)

    # compute pairwise ewma correlation
    corr_panel = pd.ewmcorr(ret_df, halflife=corr_halflife, min_periods=corr_seed_period, pairwise=True)

    # reindex axis to maintain order of columns as pandas implicity sorts axes alphabetically
    corr_panel = corr_panel.reindex_axis(ret_df.columns, axis='major_axis')
    corr_panel = corr_panel.reindex_axis(ret_df.columns, axis='minor_axis')

    return corr_panel


def _calc_ewma_covariance_and_adjcorrelation(vol_df,
                                             corr_panel,
                                             corr_type = 'full',
                                             corr_cap = None):
    '''
    Calculate EWMA (exponentially weighted moving average) covariance and adjusted (for corr type) correlation matrix
    '''

    assets = vol_df.columns
    cov_dict = {}
    adj_corr_dict = {}

    for idx, corr in corr_panel.iteritems():
        if np.isnan(np.array(corr)).all():
            adj_corr = corr.copy()
        else:
            if corr_type == 'full':
                adj_corr = corr.copy()
            elif corr_type == 'diagonal':
                valid_corr = corr.dropna(axis=0,how='all').dropna(axis=1,how='all')
                adj_corr = pd.DataFrame(np.diag(np.diag(valid_corr)), index=valid_corr.index, columns=valid_corr.index)
                adj_corr = adj_corr.reindex(index=corr.index,columns=corr.columns)
            elif corr_type == 'average':
                adj_corr = corr.copy()
                adj_corr.values[:] = np.nanmean(corr.values[np.triu_indices_from(corr.values,1)])
                adj_corr.values[np.diag_indices_from(corr.values)] = 1.0
                adj_corr.values[np.isnan(corr.values)] = np.nan
            elif corr_type == 'shrinktozero':
                adj_corr = corr.copy()
                adj_corr.values[:] = 0.5 * np.array(corr) + 0.5 * np.diag(np.diag(corr))
            elif corr_type == 'shrinktoaverage':
                adj_corr = corr.copy()
                adj_corr.values[:] = np.nanmean(corr.values[np.triu_indices_from(corr.values,1)])
                adj_corr.values[np.diag_indices_from(corr.values)] = 1.0
                adj_corr.values[np.isnan(corr.values)] = np.nan
                adj_corr.values[:] = 0.5 * corr + 0.5 * adj_corr

        # apply correlation cap
        if corr_cap:
            diag_indices = np.diag_indices_from(adj_corr)
            diagonal = adj_corr.values[diag_indices]
            adj_corr = adj_corr.clip(-corr_cap, corr_cap)
            adj_corr.values[diag_indices] = diagonal

        adj_corr_dict[idx] = adj_corr
        # only generate a valid covariance if vol array is non empty for the given period
        vol_valid = vol_df.ix[idx].dropna()
        # check the validity of corr too as corr matrix might have nans for assets where vol array might not
        # this situation arises when seed period of corr > seed period of vol
        corr_valid = adj_corr_dict[idx].dropna(axis=0,how='all').dropna(axis=1,how='all')
        if vol_valid.empty:
            cov_dict[idx] = pd.DataFrame(np.nan, index=assets, columns=assets)
        else:
            # vol_valid needs to be adjusted when seed period of corr > seed period of vol
            vol_valid = vol_valid[vol_valid.index.intersection(corr_valid.index)]
            if vol_valid.empty:
                cov_dict[idx] = pd.DataFrame(np.nan, index=assets, columns=assets)
            else:
                diag_vol_valid = pd.DataFrame(np.diag(vol_valid), index=vol_valid.index, columns=vol_valid.index)
                adj_corr_valid = adj_corr_dict[idx].ix[vol_valid.index,vol_valid.index]
                cov_valid = diag_vol_valid.dot(adj_corr_valid.dot(diag_vol_valid))
                cov_dict[idx] = pd.DataFrame(cov_valid, index=assets, columns=assets)

    return pd.Panel(cov_dict), pd.Panel(adj_corr_dict)

def calc_ewma_riskmodel(ret_df,
                        vol_halflife = 252,
                        vol_seed_period = 252,
                        var_annualization_factor = 1,
                        corr_halflife = 252,
                        corr_seed_period = 252,
                        corr_type = 'full',
                        corr_cap = None,
                        lag = 1):
    '''
    Calculate EWMA (exponentially weighted moving average) risk model

    Parameters
    ----------
    ret_df : DataFrame
        DataFrame containing asset returns (assets as columns and time as index)
    vol_halflife : int, optional
        Half-life period (periodicity determined by index of returns) for computing volatility
    vol_seed_period : int, optional
        Seeding period (periodicity determined by index of returns) for computing volatility
    var_annualization_factor : int, optional
        Factor used to annualize variance (defaults to 1, assuming daily data and 252 days (periods) to calculate variance)
    corr_halflife : int, optional
        Half-life period (periodicity determined by index of returns) for computing correlation
    corr_seed_period : int, optional
        Seeding period (periodicity determined by index of returns) for computing correlation
    corr_type : string, optional
        Specify the type of correlation matrix. Options include
        'full' : full correlation matrix (default)
        'diagonal' : diagonal correlation matrix
        'average' : off-diagonal correlations are replaced by their average
        'shrinktozero' : off-diagonal correlations are shrunk to zero
        'shrinktoaverage' : off-diagonal correlations are shrunk to off-diagonal average
    corr_cap : float, optional
        Absolute cap for correlations
    lag : int, optional
        Periods (periodicity determined by index of returns) by which to lag the returns before generating the risk model

    Returns
    -------
    riskmodel : dict
        Dictionary object containing the following DataFrames (vol, var) and Panels (corr, mod_corr, cov)
    '''

    if vol_halflife < 0:
        raise ValueError('%d is not a valid volatility half-life' % vol_halflife)
    if vol_seed_period < 0:
        raise ValueError('%d is not a valid volatility seed period' % vol_seed_period)
    if var_annualization_factor < 0:
        raise ValueError('%d is not a valid volatility annualization factor' % var_annualization_factor)
    if corr_halflife < 0:
        raise ValueError('%d is not a valid correlation half-life' % corr_halflife)
    if corr_seed_period < 0:
        raise ValueError('%d is not a valid correlation seed period' % corr_seed_period)
    if corr_cap is not None and (corr_cap < 0 or corr_cap > 1):
        raise ValueError('%f is not a valid correlation cap' % corr_cap)
    if corr_type not in ['full', 'diagonal', 'average', 'shrinktozero', 'shrinktoaverage']:
        raise ValueError('%s is not a valid correlation matrix type' % corr_type)
    if lag < 0:
        raise ValueError('%d is not a valid lag period' % lag)

    riskmodel_dict = {}
    riskmodel_dict['var'] = _calc_ewma_variance(ret_df,
                                                vol_halflife=vol_halflife,
                                                vol_seed_period=vol_seed_period,
                                                var_annualization_factor=var_annualization_factor,
                                                lag=lag)

    riskmodel_dict['vol'] = _calc_ewma_volatility(ret_df,
                                                  vol_halflife=vol_halflife,
                                                  vol_seed_period=vol_seed_period,
                                                  var_annualization_factor=var_annualization_factor,
                                                  lag=lag)

    riskmodel_dict['corr'] = _calc_ewma_correlation(ret_df,
                                                    corr_halflife=corr_halflife,
                                                    corr_seed_period=corr_seed_period,
                                                    lag=lag)

    # compute ewma covariance and adjusted correlation matrix (to reflect the corr_type and corr_cap)
    riskmodel_dict['cov'], riskmodel_dict['adj_corr'] = _calc_ewma_covariance_and_adjcorrelation(vol_df=riskmodel_dict['vol'],
                                                                                                 corr_panel=riskmodel_dict['corr'],
                                                                                                 corr_type=corr_type,
                                                                                                 corr_cap=corr_cap)

    return riskmodel_dict

def calc_ewma_riskmodel_using_impliedvol(ret_df,
                                         impvol_df,
                                         impvol_scale = 1/(np.sqrt(252)*100),
                                         corr_halflife = 252,
                                         corr_seed_period = 252,
                                         corr_type = 'full',
                                         corr_cap = None,
                                         lag = 1):
    '''
    Calculate EWMA (exponentially weighted moving average) risk model

    Parameters
    ----------
    ret_df : DataFrame
        DataFrame containing asset returns (assets as columns and time as index)
    impvol_df : DataFrame
        DataFrame containing asset implied volatility (assets as columns and time as index)        
    impvol_scale : float, optional
        Factor used to convert annualized implied vol to daily vol (defaults to 1/(np.sqrt(252)*100), assuming 252 days (periods) per year)
    corr_halflife : int, optional
        Half-life period (periodicity determined by index of returns) for computing correlation
    corr_seed_period : int, optional
        Seeding period (periodicity determined by index of returns) for computing correlation
    corr_type : string, optional
        Specify the type of correlation matrix. Options include
        'full' : full correlation matrix (default)
        'diagonal' : diagonal correlation matrix
        'average' : off-diagonal correlations are replaced by their average
        'shrinktozero' : off-diagonal correlations are shrunk to zero
        'shrinktoaverage' : off-diagonal correlations are shrunk to off-diagonal average
    corr_cap : float, optional
        Absolute cap for correlations
    lag : int, optional
        Periods (periodicity determined by index of returns) by which to lag the returns before generating the risk model

    Returns
    -------
    riskmodel : dict
        Dictionary object containing the following DataFrames (vol, var) and Panels (corr, mod_corr, cov)
    '''

    if impvol_scale < 0:
        raise ValueError('%d is not a valid implied volatility scale factor' % impvol_scale)
    if corr_halflife < 0:
        raise ValueError('%d is not a valid correlation half-life' % corr_halflife)
    if corr_seed_period < 0:
        raise ValueError('%d is not a valid correlation seed period' % corr_seed_period)
    if corr_cap is not None and (corr_cap < 0 or corr_cap > 1):
        raise ValueError('%f is not a valid correlation cap' % corr_cap)
    if corr_type not in ['full', 'diagonal', 'average', 'shrinktozero', 'shrinktoaverage']:
        raise ValueError('%s is not a valid correlation matrix type' % corr_type)
    if lag < 0:
        raise ValueError('%d is not a valid lag period' % lag)
    if not ret_df.index.equals(impvol_df.index):
        raise ValueError('The index of ret_df does not match the index of impvol_df')
    if not ret_df.columns.equals(impvol_df.columns):
        raise ValueError('ret_df and impvol_df should have the same assets')        
        
    riskmodel_dict = {}
    riskmodel_dict['vol'] = impvol_df * impvol_scale
    riskmodel_dict['var'] = riskmodel_dict['vol']**2

    riskmodel_dict['corr'] = _calc_ewma_correlation(ret_df,
                                                    corr_halflife=corr_halflife,
                                                    corr_seed_period=corr_seed_period,
                                                    lag=lag)

    # compute ewma covariance and adjusted correlation matrix (to reflect the corr_type and corr_cap)
    riskmodel_dict['cov'], riskmodel_dict['adj_corr'] = _calc_ewma_covariance_and_adjcorrelation(vol_df=riskmodel_dict['vol'],
                                                                                                 corr_panel=riskmodel_dict['corr'],
                                                                                                 corr_type=corr_type,
                                                                                                 corr_cap=corr_cap)

    return riskmodel_dict
    
def calc_HS_VaR(ret_df, 
                window=504, 
                min_periods=None,
                est_prob=0.01, 
                PV=1):
    '''
    Calculate Historial Simulation (HS) Value-at-Risk (VaR)

    Parameters
    ----------
    ret_df : DataFrame
        Asset or Portfolio returns
    window : int, optional
        Window used to compute VaR
    min_periods: int, optional
        Minimum number of periods in the window to compute VaR
    est_prob : float, optional
        VaR estimation probability (defaults to 1%)
    PV : float, optional
        Portfolio value or notional (defaults to 1)

    Returns
    -------
    HS_VaR_df : DataFrame
        Historical Simulation Value-at-Risk
    '''
    if window < 0:
        raise ValueError('%d is not a valid window size' % window)
    if est_prob < 0 or est_prob > 1:
        raise ValueError('%f is not a valid estimation probability' % est_prob)
    if PV < 0:
        raise ValueError('%f is not a valid portfolio value' % PV)
        
    HS_VaR_df = pd.rolling_quantile(ret_df, window, est_prob, min_periods=min_periods) * PV
    col = 'HS VaR (' + str(round(window/252)) + 'Y window, ' + str(est_prob*100) + '% probability' + ')'
    HS_VaR_df = HS_VaR_df.rename(columns={'Portfolio':col})
    
    return HS_VaR_df
    
def calc_Gaussian_VaR(ret_df, 
                      window=504, 
                      min_periods=None,
                      est_prob=0.01, 
                      PV=1):
    '''
    Calculate Gaussian Value-at-Risk (VaR)

    Parameters
    ----------
    ret_df : DataFrame
        Asset or Portfolio returns
    window : int, optional
        Window used to compute VaR
    min_periods: int, optional
        Minimum number of periods in the window to compute VaR
    est_prob : float, optional
        VaR estimation probability (defaults to 1%)
    PV : float, optional
        Portfolio value or notional (defaults to 1)

    Returns
    -------
    GS_VaR_df : DataFrame
        Gaussian Value-at-Risk
    '''
    if window < 0:
        raise ValueError('%d is not a valid window size' % window)
    if est_prob < 0 or est_prob > 1:
        raise ValueError('%f is not a valid estimation probability' % est_prob)
    if PV < 0:
        raise ValueError('%f is not a valid portfolio value' % PV)
        
    z = norm.ppf(est_prob)
    mu = pd.rolling_mean(ret_df, window, min_periods=min_periods)
    sigma = pd.rolling_std(ret_df, window, min_periods=min_periods)
        
    GS_VaR_df = (mu + sigma * z) * PV
    col = 'Gaussian VaR (' + str(round(window/252)) + 'Y window, ' + str(est_prob*100) + '% probability' + ')'
    GS_VaR_df = GS_VaR_df.rename(columns={'Portfolio':col})

    return GS_VaR_df