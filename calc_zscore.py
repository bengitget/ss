# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 17:36:40 2016

@author: e612727
"""

# Copyright (C) 2015 State Street Global Advisors

import pandas as pd
import numpy as np


def calc_zscore(df,
                mean_halflife=21,
                mean_seed_period=21,
                std_halflife=21,
                std_seed_period=21,
                smth_halflife=0,
                ewm=True,
                subtract_mean=True,
                cap=3.0,
                lag=0):
    """
    Calculate timeseries z-score (assuming normal distribution of input data)

    Parameters
    ----------
    df : DataFrame or Series
        DataFrame or Series object containing timeseries data
    mean_halflife : int, optional
        Half-life period (periodicity determined by index of df) for computing mean
    mean_seed_period : int, optional
        Seeding period (periodicity determined by index of df) for computing mean
    std_halflife : int, optional
        Half-life period (periodicity determined by index of df) for computing standard deviation
    std_seed_period : int, optional
        Seeding period (periodicity determined by index of df) for computing standard deviation
    smth_halflife : int, optional
        Smoothing half-life period (periodicity determined by index of df) for smoothing input data before computing z-score
    ewm : bool, optional
        If True, compute z-score based on ewm mean and standard deviation. If False, compute z-score based on simple mean and standard deviation.
    subtract_mean : bool, optional
        If True, subtract mean while computing z-score. If False, normalize the value by dividing by standard deviation.
    cap : float, optional
        Absolute cap for z-score
    lag : int, optional
        Periods (periodicity determined by index of df) by which to lag the z-score

    Returns
    -------
    score_df : DataFrame or Series
        DataFrame or Series object containing z-score

    """

    is_series = False
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
        is_series = True
    elif not isinstance(df, pd.DataFrame):
        raise ValueError('df should be either a DataFrame or Series object')

    if mean_halflife < 0:
        raise ValueError('%d is not a valid mean half-life' % mean_halflife)
    if mean_halflife > df.shape[0]:
        raise ValueError('mean_halflife can not be larger than length of index of df')
    if mean_seed_period < 0:
        raise ValueError('%d is not a valid mean seed period' % mean_seed_period)
    if mean_seed_period > df.shape[0]:
        raise ValueError('mean_seed_period can not be larger than length of index of df')
    if std_halflife < 0:
        raise ValueError('%d is not a valid standard deviation half-life' % std_halflife)
    if std_halflife > df.shape[0]:
        raise ValueError('std_halflife can not be larger than length of index of df')
    if std_seed_period < 0:
        raise ValueError('%d is not a valid standard deviation seed period' % std_seed_period)
    if std_seed_period > df.shape[0]:
        raise ValueError('std_seed_period can not be larger than length of index of df')
    if smth_halflife < 0:
        raise ValueError('%d is not a valid smoothing half-life' % smth_halflife)
    if smth_halflife > df.shape[0]:
        raise ValueError('smth_halflife can not be larger than length of index of df')
    if not isinstance(ewm, bool):
        raise ValueError('ewm should be either True of False')
    if not isinstance(subtract_mean, bool):
        raise ValueError('subtract_mean should be either True of False')
    if cap <= 0:
        raise ValueError('%f is not a valid score cap' % cap)
    if lag < 0:
        raise ValueError('%d is not a valid lag period' % lag)
    if lag > df.shape[0]:
        raise ValueError('lag can not be larger than length of index of df')

    # apply smoothing
    if smth_halflife > 0:
        df = pd.ewma(df, halflife=smth_halflife, min_periods=smth_halflife, adjust=False)

    # compute mean and standard deviation
    if ewm:
        mean_df = pd.ewma(df, halflife=mean_halflife, min_periods=mean_seed_period, adjust=False)
        std_df = pd.ewmstd(df, halflife=std_halflife, min_periods=std_seed_period, adjust=False)
    else:
        mean_df = pd.rolling_mean(df, window=mean_halflife, min_periods=mean_seed_period)
        std_df = pd.rolling_std(df, window=std_halflife, min_periods=std_seed_period)

    # compute score
    if subtract_mean:
        score_df = (df - mean_df) / std_df
    else:
        score_df = df / std_df

    # cap score
    score_df = score_df.clip(-cap, cap)

    # lag score
    if lag > 0:
        score_df = score_df.shift(lag)

    if is_series:
        return pd.Series(score_df.squeeze())
    else:
        return score_df


def score_to_alpha(score_df,
                   vol_df,
                   IC=0.1):
    """
    Compute signal alphas, given scores and IC

    Parameters
    ----------
    score_df : DataFrame
        DataFrame containing signal scores for assets
    vol_df : DataFrame
        DataFrame containing asset volatilities
    IC : float, optional
        Information Co-efficient (IC) of the signal

    Returns
    -------
    alpha_df : DataFrame
        DataFrame containing signal alphas for assets

    """

    if not isinstance(score_df, pd.DataFrame):
        raise ValueError('score_df should be a DataFrame object')
    if not isinstance(vol_df, pd.DataFrame):
        raise ValueError('vol_df should be a DataFrame object')
    if IC <= 0:
        raise ValueError('%d is not a valid IC' % IC)

    if not score_df.index.equals(vol_df.index):
        raise ValueError('score_df and vol_df should have the same index')
    if not score_df.columns.equals(vol_df.columns):
        raise ValueError('score_df and vol_df should have the same columns')

    return score_df * vol_df * IC


def calc_xscore(df,
                smth_halflife=0,
                min_observations=2,
                subtract_mean=True,
                cap=3.0,
                lag=0):
    """
    Calculate cross-sectional score (x-score)

    Parameters
    ----------
    df : DataFrame
        DataFrame object containing timeseries data
    smth_halflife : int, optional
        Smoothing half-life period (periodicity determined by index of df) for smoothing input data before computing x-score
    min_observations : int, optional
        Minimum number of cross-sectional data points required for computing score
    subtract_mean : bool, optional
        If True, subtract cross-sectional mean while computing x-score. If False, normalize the value by dividing by standard deviation of cross-section.
    cap : float, optional
        Absolute cap for x-score
    lag : int, optional
        Periods (periodicity determined by index of df) by which to lag the x-score

    Returns
    -------
    score_df : DataFrame
        DataFrame object containing x-score

    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError('df should be a DataFrame object')

    if smth_halflife < 0:
        raise ValueError('%d is not a valid smoothing half-life' % smth_halflife)
    if smth_halflife > df.shape[0]:
        raise ValueError('smth_halflife can not be larger than length of index of df')
    if min_observations < 2:
        raise ValueError('%d is not a valid number of minimum observations' % min_observations)
    if min_observations > df.shape[1]:
        raise ValueError('min_observations can not be greater than the number of columns of df')
    if not isinstance(subtract_mean, bool):
        raise ValueError('subtract_mean should be either True of False')
    if cap <= 0:
        raise ValueError('%f is not a valid score cap' % cap)
    if lag < 0:
        raise ValueError('%d is not a valid lag period' % lag)
    if lag > df.shape[0]:
        raise ValueError('lag can not be larger than length of index of df')

    # apply min observations filter
    df[df.count(axis=1) < min_observations] = np.nan

    # apply smoothing
    if smth_halflife > 0:
        df = pd.ewma(df, halflife=smth_halflife, min_periods=smth_halflife, adjust=False)

    # compute score
    if subtract_mean:
        score_df = (df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1), axis=0)
    else:
        score_df = df.div(df.std(axis=1), axis=0)

    # cap score
    score_df = score_df.clip(-cap, cap)

    # lag score
    if lag > 0:
        score_df = score_df.shift(lag)

    return score_df

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
