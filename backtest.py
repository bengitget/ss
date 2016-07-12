# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 17:36:01 2016

@author: e612727
"""

# Copyright (C) 2015 State Street Global Advisors

import pandas as pd
import numpy as np
import math
import calendar
import scipy.stats
import statsmodels.tsa.stattools as stattools
from datetime import time
#from eureka.risk import calc_HS_VaR, calc_Gaussian_VaR, _calc_ewma_volatility
#from eureka.optimize import mean_variance_optimizer

def _reshape_tcosts(linear_tcosts, reference_df):
    """
    Reshape transaction costs object into a DataFrame
    """
    if linear_tcosts is None:
        linear_tcosts = 0.0

    if isinstance(linear_tcosts, pd.DataFrame):
        if linear_tcosts.isnull().any().any():
            linear_tcosts = linear_tcosts.fillna(0.0)
        if not linear_tcosts.index.equals(reference_df.index):
            raise ValueError('linear_tcosts and reference_df should have the same date index')
        elif not linear_tcosts.columns.equals(reference_df.columns):
            raise ValueError('linear_tcosts and reference_df should have the same assets')
        else:
            linear_tcosts_df = linear_tcosts.reindex(columns=reference_df.columns)
    elif isinstance(linear_tcosts, pd.Series):
        if linear_tcosts.isnull().any():
            linear_tcosts = linear_tcosts.fillna(0.0)
        if not linear_tcosts.index.equals(reference_df.columns):
            raise ValueError('linear_tcosts does not have transaction costs for all assets')
        else:
            linear_tcosts = linear_tcosts.reindex(index=reference_df.columns)
            linear_tcosts_df = pd.DataFrame(np.nan, index=reference_df.index, columns=reference_df.columns)
            linear_tcosts_df.ix[0] = linear_tcosts
            linear_tcosts_df = linear_tcosts_df.ffill()
    elif not isinstance(linear_tcosts, float):
        raise TypeError('linear_tcosts should be a float, Series or DataFrame object')
    else:
        linear_tcosts_df = pd.DataFrame(linear_tcosts, index=reference_df.index, columns=reference_df.columns)
    return linear_tcosts_df

def _calc_return_stats(ret_ds):
    """
    Calculate return statistics
    """
    stats = {}
    stats['mean'] = ret_ds.mean()
    stats['std'] = ret_ds.std()
    stats['skew'] = ret_ds.skew()
    stats['kurt'] = ret_ds.kurt()
    stats['count'] = ret_ds.count()
    stats['tstat'] = stats['mean'] / stats['std'] * np.sqrt(stats['count'])
    return pd.Series(stats)

def calc_IR(va_df, annualization_factor=252):
    """
    Calculate IR (Information Ratio)

    Parameters
    ----------
    va_df : DataFrame or Series
        DataFrame or Series containing value added returns
    annualization_factor : int, optional
        Factor used to annualize IR (defaults to 252, assuming daily data and 252 days (periods) to calculate risk and
        returns)

    Returns
    -------
    IR : Series or float
        Information Ratio

    """
    if annualization_factor <= 0:
        raise ValueError('%d is not a valid annualization factor' % annualization_factor)
    IR = np.sqrt(annualization_factor) * va_df.mean()/va_df.std()
    return IR

def calc_IR_rolling(va_df, annualization_factor=252, window=252):
    """
    Calculate rolling IR

    Parameters
    ----------
    va_df : DataFrame or Series
        DataFrame or Series containing value added returns
    annualization_factor : int, optional
        Factor used to annualize IR (defaults to 252, assuming daily data)
    window : int, optional
        Window for rolling (defaults to 252).

    Returns
    -------
    IR_rolling : Series or DataFrame
        rolling IR
    """
    IR_rolling = pd.rolling_mean(va_df, window) * np.sqrt(annualization_factor) / pd.rolling_std(va_df, window)
    return IR_rolling
        
def _calc_time_aggregate_IR_for_series(va_ds, annualization_factor=252, skip_first_order_acf=True):
    weight = np.arange(annualization_factor-1, 0, -1)
    # compute acf of the return series
    acf = stattools.acf(va_ds.fillna(0), nlags=(annualization_factor-1))
    # eliminate lag-0 auto correlation
    acf = acf[1:]
    # eliminate the first-order AR coefficient if it is large and negative (i.e. AR(1) process with negative beta)
    if skip_first_order_acf:
        if acf[0] < 0 and acf[0] == min(acf):
            acf[0] = 0
    # compute the IR adjustment factor
    adjustment_factor = annualization_factor / np.sqrt(annualization_factor + 2 * np.dot(weight, acf))
    return adjustment_factor * va_ds.mean()/va_ds.std()

def calc_time_aggregate_IR(va_df, annualization_factor=252, skip_first_order_acf=True):
    """
    Calculate time-aggregate IR (Information Ratio)

    Parameters
    ----------
    va_df : DataFrame or Series
        DataFrame or Series containing value added returns
    annualization_factor : int, optional
        Factor used to annualize IR (defaults to 252, assuming daily data and 252 days (periods) to calculate risk and
        returns)
    skip_first_order_acf : bool, optional
        If True, skip the first-order auto-correlation coefficient (AR(1)) in computing the adjustment factor. If False, include the first-order ACF.

    Returns
    -------
    taIR : Series or float
        Time-aggregate Information Ratio

    Reference
    ---------
    Equation 20 in "The Statistics of Sharpe Ratios" by Andrew Lo (2002)

    Notes
    -----
    The most common method for performing time aggregation of IR (e.g. annual) is to multiply the higher-frequency
    IR (e.g. daily) by the square root of the number of periods contained in the lower-frequency holding period
    (e.g. multiply a daily estimator by sqrt(252) to obtain the annual estimator). This rule of thumb is correct
    only under the assumption of IID returns. For non-IID returns, an alternate procedure that accounts for serial
    correlation in returns is requried.

    For non-IID Returns, the relationship between daily IR and annual IR is somewhat more involved because the
    variance of annual returns is not just the sum of the variances of daily returns but also includes all the covariances.
    Specifically, under the assumption that returns are stationary:

    annual_IR = adjustment_factor * daily_IR

    where adjustment_factor = annualization_period / sqrt (annualization_period +  2 * summation of (k-th order ACF of returns * (annualization_period - k)))

    and k = 1 to (annualization_period - 1)

    For IID returns, k-th order ACF of returns = 0 for k ranging from 1 to (annualization_period - 1), therefore the
    adjustment_factor reduces to just sqrt(annualization_period)

    """
    if annualization_factor <= 0:
        raise ValueError('%d is not a valid annualization factor' % annualization_factor)
    if not (isinstance(va_df, pd.Series) or isinstance(va_df, pd.DataFrame)):
        raise TypeError('va_df should be either a DataFrame or Series object')
    if va_df.shape[0] <= annualization_factor:
        raise ValueError('va_df should have more elements than annualization_factor')

    if isinstance(va_df, pd.Series):
        return _calc_time_aggregate_IR_for_series(va_df, annualization_factor, skip_first_order_acf)
    elif isinstance(va_df, pd.DataFrame):
        taIR = pd.Series(index=va_df.columns)
        for idx in va_df.columns:
            taIR[idx] = _calc_time_aggregate_IR_for_series(va_df[idx], annualization_factor, skip_first_order_acf)
        return taIR

def calc_value_added(ret_df,
                     holdings_df,
                     linear_tcosts=None,
                     cumulate=False,
                     cumulate_method='logsum',
                     portfolio=False,
                     portfolio_name='Portfolio'):
    """
    Calculate value-added (VA) returns using the convention that holdings for a given period are already aligned with
    returns.

    Parameters
    ----------
    ret_df : DataFrame
        DataFrame containing asset returns (assets as columns and time as index)
    holdings_df : DataFrame
        DataFrame containing signal holdings for assets
    linear_tcosts : float, Series or DataFrame, optional
        Linear transaction costs are applicable for unit change in asset holdings
        If linear_tcosts is float, the same linear transaction costs are applied to all assets
        If linear_tcosts is a Series object, the vector specifies the transaction costs applicable for each asset
        If linear_tcosts is a DataFrame object, then the DataFrame specifies transaction costs applicable for each asset
        (column) for each period (index)
    cumulate : bool, optional
        If True, calculate the cumulative value-added returns. If False, just compute the value-added returns.
    cumulate_method : str, optional
        Method used to cumulate value added returns
    portfolio : bool, optional
        If True, calculate the cumulative value-added returns for the portfolio. If False, just compute the value-added returns for assets.
    portfolio_name : str, optional
        Name given to the portfolio

    Returns
    -------
    va_df : DataFrame
        Value-added returns

    """
    if not isinstance(ret_df, pd.DataFrame):
        raise TypeError('ret_df should be a DataFrame object')
    if not isinstance(holdings_df, pd.DataFrame):
        raise TypeError('holdings_df should be a DataFrame object')

    if not ret_df.index.equals(holdings_df.index):
        raise ValueError('ret_df and holdings_df should have the same date index')
    if not ret_df.columns.equals(holdings_df.columns):
        raise ValueError('ret_df and holdings_df should have the same assets')

    linear_tcosts_df = _reshape_tcosts(linear_tcosts, holdings_df)

    abs_trades_df = holdings_df.diff().abs().fillna(0.0)
    va_df = ret_df * holdings_df - abs_trades_df * linear_tcosts_df

    if portfolio:
        va_df = pd.DataFrame(va_df.sum(axis=1), index=va_df.index, columns=[portfolio_name])

    if cumulate:
        if cumulate_method == 'sum':
            va_df = va_df.cumsum()
        elif cumulate_method == 'prod':
            va_df = (1 + va_df).cumprod() - 1
        elif cumulate_method == 'logsum':
            va_df = np.log(1 + va_df).cumsum()
        else:
            raise ValueError('%s is not a valid cumulate method' % cumulate_method)

    return va_df

def calc_monthly_value_added(va_df,
                            cumulate_method='logsum'):
    """
    Calculate monthly value-added returns given daily series

    Parameters
    ----------
    va_df : DataFrame
        Daily value-added returns
    cumulate_method : str, optional
        Method used to cumulate value added returns

    Returns
    -------
    df : DataFrame
        A table of VAs by month and year
    """  
    if not isinstance(va_df, pd.DataFrame):
        raise TypeError('ret_df should be a DataFrame object')
        
    # cumulate function
    if cumulate_method == 'sum':
        cva_func = lambda va: va.cumsum().iloc[-1]
    elif cumulate_method == 'prod':
        cva_func = lambda va: (1 + va).cumprod().iloc[-1] - 1
    elif cumulate_method == 'logsum':
        cva_func = lambda va: np.log(1 + va).cumsum().iloc[-1]
    else:
        raise ValueError('%s is not a valid cumulate method' % cumulate_method)
        
    calendar_months = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                       7: 'Jul', 8: 'Aug', 9: 'Sep',10: 'Oct',11: 'Nov',12: 'Dec'}

    df = {}
    for idx in calendar_months.keys():
        df[calendar_months[idx]] = {}

    # compute monthly VA
    vagp = va_df.groupby(lambda x: (x.year, x.month))
    monthly_va = vagp.aggregate(cva_func)
    for (y, m), val in monthly_va.iterrows():
        df[calendar_months[m]][y] = val[0] * 100.0
    
    # compute annual VA
    vagp = va_df.groupby(lambda x: x.year)
    annual_va = vagp.aggregate(cva_func)
    df['Year'] = {}
    for y, val in annual_va.iterrows():
        df['Year'][y] = val[0] * 100.0
        
    columns = list(calendar_months.values()) + ['Year']
    df = pd.DataFrame(df, columns=columns)

    return df

def calc_off_the_top_IR(ret_df,
                        holdings_df,
                        linear_tcosts=None,
                        annualization_factor=252):
    """
    Calculate off-the-top IR (Information Ratio)

    Parameters
    ----------
    ret_df : DataFrame
        DataFrame containing asset returns (assets as columns and time as index)
    holdings_df : DataFrame
        DataFrame containing signal holdings for assets
    linear_tcosts : float, Series or DataFrame, optional
        Linear transaction costs are applicable for unit change in asset holdings
        If linear_tcosts is float, the same linear transaction costs are applied to all assets
        If linear_tcosts is a Series object, the vector specifies the transaction costs applicable for each asset
        If linear_tcosts is a DataFrame object, then the DataFrame specifies transaction costs applicable for each asset
        (column) for each period (index)
    annualization_factor : int, optional
        Factor used to annualize IR (defaults to 252, assuming daily data and 252 days (periods) to calculate risk and returns)

    Returns
    -------
    ottIR_ds : Series
        Off-the-top IRs

    """
    if not isinstance(ret_df, pd.DataFrame):
        raise TypeError('ret_df should be a DataFrame object')
    if not isinstance(holdings_df, pd.DataFrame):
        raise TypeError('holdings_df should be a DataFrame object')
    if annualization_factor <= 0:
        raise ValueError('%d is not a valid annualization factor' % annualization_factor)

    if not ret_df.index.equals(holdings_df.index):
        raise ValueError('ret_df and holdings_df should have the same date index')
    if not ret_df.columns.equals(holdings_df.columns):
        raise ValueError('ret_df and holdings_df should have the same assets')

    linear_tcosts_df = _reshape_tcosts(linear_tcosts, holdings_df)

    ottIR_ds = pd.Series(index=ret_df.columns)
    for idx in ret_df.columns:
        ott_portfolio_va_df = calc_value_added(ret_df.drop(idx, axis=1), holdings_df.drop(idx, axis=1),
                                               linear_tcosts=linear_tcosts_df.drop(idx, axis=1), portfolio=True)
        ottIR_ds[idx] = calc_IR(ott_portfolio_va_df, annualization_factor=annualization_factor)

    IR_ds = calc_IR(calc_value_added(ret_df, holdings_df, linear_tcosts=linear_tcosts_df, portfolio=True),
                    annualization_factor=annualization_factor)
    IR_ds = IR_ds.rename(index={'Portfolio': 'ALL'})
    ottIR_ds = ottIR_ds.append(IR_ds)
    #sort by values
    ottIR_ds = ottIR_ds.sort_values()
    return ottIR_ds


def calc_lead_lag_IR(ret_df,
                     holdings_df,
                     linear_tcosts=None,
                     annualization_factor=252,
                     lead_lag_period=21):
    """
    Calculate lead/lag IR (Information Ratio)

    Parameters
    ----------
    ret_df : DataFrame
        DataFrame containing asset returns (assets as columns and time as index)
    holdings_df : DataFrame
        DataFrame containing signal holdings for assets
    linear_tcosts : float, Series or DataFrame, optional
        Linear transaction costs are applicable for unit change in asset holdings
        If linear_tcosts is float, the same linear transaction costs are applied to all assets
        If linear_tcosts is a Series object, the vector specifies the transaction costs applicable for each asset
        If linear_tcosts is a DataFrame object, then the DataFrame specifies transaction costs applicable for each asset (column) for each period (index)
    annualization_factor : int, optional
        Factor used to annualize IR (defaults to 252, assuming daily data and 252 days (periods) to calculate risk and returns)
    lead_lag_period : int, optional
        Period used to compute lead/lag IR (defaults to 21, assuming daily data and one month (21 days) range to compute lead/lag IR)

    Returns
    -------
    llIR_ds : Series
        Lead/lag IRs

    """
    if not isinstance(ret_df, pd.DataFrame):
        raise TypeError('ret_df should be a DataFrame object')
    if not isinstance(holdings_df, pd.DataFrame):
        raise TypeError('holdings_df should be a DataFrame object')
    if annualization_factor <= 0:
        raise ValueError('%d is not a valid annualization factor' % annualization_factor)
    if lead_lag_period <= 0:
        raise ValueError('%d is not a valid lead lag period' % lead_lag_period)

    if not ret_df.index.equals(holdings_df.index):
        raise ValueError('ret_df and holdings_df should have the same date index')
    if not ret_df.columns.equals(holdings_df.columns):
        raise ValueError('ret_df and holdings_df should have the same assets')

    linear_tcosts_df = _reshape_tcosts(linear_tcosts, holdings_df)

    llIR_ds = pd.Series(index=range(-lead_lag_period,lead_lag_period+1))
    for idx in range(-lead_lag_period,lead_lag_period+1):
        ll_portfolio_va_df = calc_value_added(ret_df, holdings_df.shift(idx), linear_tcosts=linear_tcosts_df.shift(idx), portfolio=True)
        llIR_ds[idx] = calc_IR(ll_portfolio_va_df, annualization_factor=annualization_factor)

    return llIR_ds

def calc_fx_30min_fix_lead_lag(ret_panel,
                               holdings_df,
                               end_time=time(17,0),
                               linear_tcosts=None,
                               annualization_factor=252,
                               lead_lag_period=21):
    """
    Calculate lead/lag IR based on FX 30-mix FIX returns

    Parameters
    ----------
    ret_panel : Panel
        Panel containing fx 30-min FIX returns (items are FIX times, major_axis is dates and minor_axis is FX assets)
    holdings_df : DataFrame
        DataFrame containing signal holdings for assets
    linear_tcosts : float, Series or DataFrame, optional
        Linear transaction costs are applicable for unit change in asset holdings
        If linear_tcosts is float, the same linear transaction costs are applied to all assets
        If linear_tcosts is a Series object, the vector specifies the transaction costs applicable for each asset
        If linear_tcosts is a DataFrame object, then the DataFrame specifies transaction costs applicable for each asset (column) for each period (index)
    annualization_factor : int, optional
        Factor used to annualize IR (defaults to 252, assuming daily data and 252 days (periods) to calculate risk and returns)
    lead_lag_period : int, optional
        Period used to compute lead/lag IR (defaults to 21, assuming daily data and one month (21 days) range to compute lead/lag IR)

    Returns
    -------
    lead_lag_df : DataFrame
        Lead/lag IRs for FIX times

    """
    if not isinstance(ret_panel, pd.Panel):
        raise TypeError('ret_panel should be a Panel object')
    if not isinstance(holdings_df, pd.DataFrame):
        raise TypeError('holdings_df should be a DataFrame object')
    if annualization_factor <= 0:
        raise ValueError('%d is not a valid annualization factor' % annualization_factor)
    if lead_lag_period <= 0:
        raise ValueError('%d is not a valid lead lag period' % lead_lag_period)

    if not ret_panel.minor_axis.equals(holdings_df.columns):
        raise ValueError('ret_panel and holdings_df should have the same assets')

    holdings_df = holdings_df.reindex(index=ret_panel.major_axis)

    lead_lag_df = {}
    for contract_time, rdf in ret_panel.iteritems():
        lead_lag_df[contract_time] = calc_lead_lag_IR(rdf, holdings_df,
                                                      linear_tcosts=linear_tcosts,
                                                      annualization_factor=annualization_factor,
                                                      lead_lag_period=lead_lag_period)
    lead_lag_df = pd.DataFrame(lead_lag_df)
    lead_lag_df = pd.concat([lead_lag_df.ix[:,lead_lag_df.columns > end_time],
                             lead_lag_df.ix[:,lead_lag_df.columns <= end_time]], axis=1)
    return lead_lag_df

def calc_realized_IC(ret_df,
                     alpha_df,
                     vol_df):
    """
    Calculate realized IC (Information Coefficient)

    Parameters
    ----------
    ret_df : DataFrame
        DataFrame containing asset returns (assets as columns and time as index)
    alpha_df : DataFrame
        DataFrame containing signal alphas for assets
    vol_df : DataFrame
        DataFrame containing asset volatilities

    Returns
    -------
    IC_df : Series
        Realized IC (defined as pairwise correlation (Spearman's rank correlation coefficient) between vol-adjusted returns and vol-adjusted alpha)
    pval_df : Series
        two-tailed p-value to test for non-correlation (roughly indicates the probability of an uncorrelated system producing datasets that have a Spearman correlation at least as extreme as the one computed from the given datasets)

    """
    if not isinstance(ret_df, pd.DataFrame):
        raise TypeError('ret_df should be a DataFrame object')
    if not isinstance(alpha_df, pd.DataFrame):
        raise ValueError('alpha_df should be a DataFrame object')
    if not isinstance(vol_df, pd.DataFrame):
        raise ValueError('vol_df should be a DataFrame object')

    if not ret_df.index.equals(alpha_df.index):
        raise ValueError('ret_df and alpha_df should have the same date index')
    if not ret_df.columns.equals(alpha_df.columns):
        raise ValueError('ret_df and alpha_df should have the same assets')
    if not ret_df.index.equals(vol_df.index):
        raise ValueError('ret_df and vol_df should have the same date index')
    if not ret_df.columns.equals(vol_df.columns):
        raise ValueError('ret_df and vol_df should have the same assets')

    IC_df = pd.DataFrame(0.0, columns=['IC'], index=ret_df.columns)
    pval_df = pd.DataFrame(0.0, columns=['PVAL'], index=ret_df.columns)

    for idx in ret_df.columns:
        vol_adj_ret_ds = ret_df[idx] / vol_df[idx]
        vol_adj_alpha_ds = alpha_df[idx] / vol_df[idx]
        df = pd.DataFrame({'return':vol_adj_ret_ds, 'alpha':vol_adj_alpha_ds}).dropna()
        if not df.empty:
            IC_df.ix[idx], pval_df.ix[idx] = scipy.stats.spearmanr(df['return'],df['alpha'])
    IC_df = IC_df.sort_values(['IC'])
    pval_df = pval_df.reindex(index=IC_df.index)
    return IC_df.squeeze(), pval_df.squeeze()


def calc_tilt_holdings(holdings_df,
                       tilt_period=252):
    """
    Calculate tilt holdings for the given tilt period

    Parameters
    ----------
    holdings_df : DataFrame
        DataFrame containing signal holdings for assets
    tilt_period : int, optional
        Period used to compute portfolio tilt (defaults to 252, assuming daily data and one year (252 days) to calculate portfolio tilt)

    Returns
    -------
    holdings_tilt_df : DataFrame
        Tilt holdings computed using the given tilt period

    """
    if tilt_period <= 0:
        raise ValueError('%d is not a valid tilt period' % tilt_period)
    if tilt_period >= holdings_df.shape[0]:
        raise ValueError('tilt period %d can not be greater than number of periods in holdings' % tilt_period)
    holdings_tilt_df = pd.rolling_mean(holdings_df, window=tilt_period, min_periods=tilt_period)
    holdings_tilt_df = holdings_tilt_df.fillna(0.0)
    return holdings_tilt_df


def calc_portfolio_performance_metrics(portfolio_va_df,
                                       holdings_df,
                                       annualization_factor=252):
    """
    Calculate portfolio performance metrics (Return, Risk, IR, Turnover etc.)

    Parameters
    ----------
    portfolio_va_df : DataFrame or Series
        DataFrame or Series containing value-added returns for portfolio
    holdings_df : DataFrame
        DataFrame containing signal holdings for assets
    annualization_factor : int, optional
        Factor used to annualize IR (defaults to 252, assuming daily data and 252 days (periods) to calculate risk and returns)

    Returns
    -------
    pmetrics_ds : Series
        Portfolio performance metrics (Return, Risk, IR, IR_firsthalf, IR_secondhalf, Turnover)

    """
    if not isinstance(portfolio_va_df, pd.DataFrame):
        raise TypeError('portfolio_va_df should be a DataFrame object')
    if not isinstance(holdings_df, pd.DataFrame):
        raise TypeError('holdings_df should be a DataFrame object')
    if annualization_factor <= 0:
        raise ValueError('%d is not a valid annualization factor' % annualization_factor)

    if not portfolio_va_df.index.equals(portfolio_va_df.index):
        raise ValueError('ret_df and holdings_df should have the same date index')

    num_of_periods = portfolio_va_df.shape[0]
    pmetrics_ds = pd.Series(np.nan, index=['Return','Risk','IR','IR_firsthalf','IR_secondhalf','Turnover'])
    pmetrics_ds['Return'] = portfolio_va_df.mean() * annualization_factor
    pmetrics_ds['Risk'] = portfolio_va_df.std() * np.sqrt(annualization_factor)
    pmetrics_ds['IR'] = calc_IR(portfolio_va_df, annualization_factor=annualization_factor)
    pmetrics_ds['IR_firsthalf'] = calc_IR(portfolio_va_df.head(math.floor(num_of_periods/2)), annualization_factor=annualization_factor)
    pmetrics_ds['IR_secondhalf'] = calc_IR(portfolio_va_df.tail(math.ceil(num_of_periods - num_of_periods/2)), annualization_factor=annualization_factor)
    pmetrics_ds['TAIR'] = calc_time_aggregate_IR(portfolio_va_df, annualization_factor=annualization_factor)['Portfolio']

    # calculate annualized turnover
    abs_trades_df = holdings_df.fillna(0).diff().abs()
    num_of_trading_periods = abs_trades_df.count().max()
    pmetrics_ds['Turnover'] = abs_trades_df.sum().sum() / num_of_trading_periods * annualization_factor

    return pmetrics_ds


def calc_IR_by_year(va_df, 
                    annualization_factor=252):
    """
    Calculate Information Ratio (IR), returns and risk by year

    Parameters
    ----------
    va_df : DataFrame
        DataFrame containing value-added timeseries data for portfolio
    annualization_factor : int, optional
        Factor used to annualize IR (defaults to 252, assuming daily data and 252 days (periods) to calculate risk and returns)

    Returns
    -------
    table_df : DataFrame
        DataFrame containing number of observations, IRs, return and risk by year
    """
    if va_df.shape[1] != 1:
        raise ValueError('va_df should have only one timeseries/column')
    if annualization_factor <= 0:
        raise ValueError('%d is not a valid annualization factor' % annualization_factor)
    grouped = va_df.groupby(lambda t: t.year)
    count_df = grouped.count().astype(int)
    count_df.columns = ['No of Observations']
    IR_df = np.round(grouped.mean() * math.sqrt(annualization_factor) / grouped.std(), 2)
    IR_df.columns = ['IR']
    return_df = np.round(grouped.mean() * annualization_factor * 100, 2)
    return_df.columns = ['Annual Returns (in %)']
    risk_df = np.round(grouped.std() * math.sqrt(annualization_factor) * 100, 2)
    risk_df.columns = ['Annual Risk (in %)']
    table_df = count_df.join(IR_df).join(return_df).join(risk_df)

    return table_df

def calc_drawdown(portfolio_va_df,
                  cumulate_method='logsum'):
    """
    Calculate drawdown (peak to trough) for the given portfolio returns

    Parameters
    ----------
    portfolio_va_df : DataFrame
        DataFrame or Series containing value-added returns for portfolio
    cumulate_method : string, optional
        Method used to cumulate value added returns

    Returns
    -------
    drawdown_df : DataFrame
        Peak to trough drawdown timeseries

    """
    # replace np.nan by 0
    portfolio_va_df[portfolio_va_df.isnull()] = 0.0
    if cumulate_method == 'sum':
        cva_df = portfolio_va_df.cumsum()
    elif cumulate_method == 'prod':
        cva_df = (1 + portfolio_va_df).cumprod() - 1
    elif cumulate_method == 'logsum':
        cva_df = np.log(1 + portfolio_va_df).cumsum()
    else:
        raise ValueError('%s is not a valid cumulate method' % cumulate_method)
   
    maxcva_df = cva_df.cummax(axis=0)
    maxcva_df = maxcva_df.clip(0,None)
    drawdown_df = cva_df - maxcva_df
    drawdown_df = drawdown_df.rename(columns={portfolio_va_df.columns[0]:'Drawdown'})
    return drawdown_df
  
def calc_drawdown_periods(drawdown_df):
    """
    Calculate drawdown periods and relevant statistics (to, from, peak to trough, trough to recovery etc.)

    Parameters
    ----------
    drawdown_df : DataFrame
        Peak to trough drawdown timeseries

    Returns
    -------
    drawdown_table_df : DataFrame
        Summary table of drawdown periods

    """
    drawdown_ds = drawdown_df.squeeze()
    drawdown_days = drawdown_ds < 0
    drawdown_table_df = pd.DataFrame(columns=['From', 'Trough', 'To', 'Depth (in %)', 'Length (in days)', 'Slump (in days)', 'Recovery (in days)'])
    
    prior_val = False
    from_dt = pd.NaT
    for dt, value in drawdown_days.iteritems():
        if value and (value != prior_val):
            from_dt = dt
        elif not value and (value != prior_val):
            section = drawdown_ds[from_dt:dt]
            trough_dt = section.idxmin()
            drawdown_table_df = drawdown_table_df.append({'From': from_dt, 
                                                          'Trough': trough_dt, 
                                                          'To': dt, 
                                                          'Depth (in %)': round(section.min() * 100, 2), 
                                                          'Length (in days)': len(section), 
                                                          'Slump (in days)': len(drawdown_ds[from_dt:trough_dt]), 
                                                          'Recovery (in days)': len(drawdown_ds[trough_dt:dt]) - 1}, 
                                                          ignore_index=True)
                
            from_dt = pd.NaT
        prior_val = value
     
    # account for the last period 
    if from_dt is not pd.NaT:
        section = drawdown_ds[from_dt:dt]
        trough_dt = section.idxmin()
        drawdown_table_df = drawdown_table_df.append({'From': from_dt, 
                                                      'Trough': trough_dt, 
                                                      'To': dt, 
                                                      'Depth (in %)': round(section.min() * 100, 2),  
                                                      'Length (in days)': len(section), 
                                                      'Slump (in days)': len(drawdown_ds[from_dt:trough_dt]), 
                                                      'Recovery (in days)': len(drawdown_ds[trough_dt:dt]) - 1}, 
                                                       ignore_index=True)
    # sort drawdown periods                                                   
    drawdown_table_df = drawdown_table_df.sort_values(['Depth (in %)'])
    return drawdown_table_df

def backtest_metrics(ret_df,
                     holdings_df,
                     riskmodel_dict=None,
                     score_df=None,
                     alpha_df=None,
                     risk_aversion=1.0,
                     tcost_aversion=1.0,
                     linear_tcosts=None,
                     annualization_factor=252,
                     tilt_period=252,
                     lead_lag_period=21,
                     rolling_turnover_period=252,
                     trailing_expost_period=126,
                     cumulate_method='logsum',
                     optimizer='mean_variance',
                     crosssectional_bound=0.001,
                     number_of_assets_max_threshold=25):
    """
    Calculate backtest metrics with transaction costs

    Parameters
    ----------
    ret_df : DataFrame
        DataFrame containing asset returns (assets as columns and time as index)
    holdings_df : DataFrame
        DataFrame containing signal holdings for assets
    riskmodel_dict : dict
        Dictionary object containing the following DataFrames (vol, var) and Panels (corr, mod_corr, cov)
    score_df : DataFrame
        DataFrame containing signal scores for assets
    alpha_df : DataFrame
        DataFrame containing signal alphas for assets
    risk_aversion : float, optional
        Risk aversion parameter is used to modulate portfolio's ex-post risk
    tcost_aversion : float, optional
        Transaction-cost aversion parameter is used to modulate aversion to transaction costs in the optimizer
    linear_tcosts : float, Series or DataFrame, optional
        Linear transaction costs are applicable for unit change in asset holdings
        If linear_tcosts is float, the same linear transaction costs are applied to all assets
        If linear_tcosts is a Series object, the vector specifies the transaction costs applicable for each asset
        If linear_tcosts is a DataFrame object, then the DataFrame specifies transaction costs applicable for each asset (column) for each period (index)
    annualization_factor : int, optional
        Factor used to annualize IR (defaults to 252, assuming daily data and 252 days (periods) to calculate risk and returns)
    tilt_period : int, optional
        Period used to compute portfolio tilt (defaults to 252, assuming daily data and one year (252 days) to calculate portfolio tilt)
    lead_lag_period : int, optional
        Period used to compute lead/lag IR (defaults to 21, assuming daily data and one month (21 days) range to compute lead/lag IR)
    rolling_turnover_period : int, optional
        Period used to compute rolling turnover (defaults to 252)
    trailing_expost_period : int, optional
        Period used to compute trailing ex-post risk (defaults to 126)
    cumulate_method : string, optional
        Method used to cumulate value added returns
    optimizer : string, optional
        Optimizer used to compute cross-sectionally neutral holdings
    crosssectional_bound : float, optional
        Lower/Upper bound on cross-sectional neutrality of holdings (default is set to 0.1%)
    number_of_assets_max_threshold : int, optional
        If the number of assets exceed the threshold, then certain metrics (off-the-top IR etc.) are not generated.

    Returns
    -------
    backtest_dict : dict
        Dictionary object containing backtest metrics

    """
    if not isinstance(ret_df, pd.DataFrame):
        raise TypeError('ret_df should be a DataFrame object')
    if not isinstance(holdings_df, pd.DataFrame):
        raise TypeError('holdings_df should be a DataFrame object')
    if riskmodel_dict is not None:
        if not isinstance(riskmodel_dict, dict):
            raise TypeError('riskmodel_dict should be a dictionary object')
    if score_df is not None:
        if not isinstance(score_df, pd.DataFrame):
            raise TypeError('score_df should be a DataFrame object')
    if alpha_df is not None:
        if not isinstance(alpha_df, pd.DataFrame):
            raise TypeError('alpha_df should be a DataFrame object')
    if risk_aversion <= 0:
        raise ValueError('%f is not a valid risk aversion' % risk_aversion)
    if tcost_aversion <= 0:
        raise ValueError('%f is not a valid tcost aversion' % tcost_aversion)
    if annualization_factor <= 0:
        raise ValueError('%d is not a valid annualization factor' % annualization_factor)
    if tilt_period <= 0:
        raise ValueError('%d is not a valid tilt period' % tilt_period)
    if lead_lag_period <= 0:
        raise ValueError('%d is not a valid lead lag period' % lead_lag_period)
    if rolling_turnover_period <= 0:
        raise ValueError('%d is not a valid rolling turnover period' % rolling_turnover_period)
    if trailing_expost_period <= 0:
        raise ValueError('%d is not a valid trailing ex-post period' % trailing_expost_period)
    if tilt_period >= holdings_df.shape[0]:
        raise ValueError('tilt period %d can not be greater than number of periods in holdings' % tilt_period)

    if not ret_df.index.equals(holdings_df.index):
        raise ValueError('ret_df and holdings_df should have the same date index')
    if not ret_df.columns.equals(holdings_df.columns):
        raise ValueError('ret_df and holdings_df should have the same assets')
    if score_df is not None:
        if not ret_df.index.equals(score_df.index):
            raise ValueError('ret_df and score_df should have the same date index')
        if not ret_df.columns.equals(score_df.columns):
            raise ValueError('ret_df and score_df should have the same assets')
    if alpha_df is not None:
        if not ret_df.index.equals(alpha_df.index):
            raise ValueError('ret_df and alpha_df should have the same date index')
        if not ret_df.columns.equals(alpha_df.columns):
            raise ValueError('ret_df and alpha_df should have the same assets')

    if riskmodel_dict is not None:
        if 'cov' not in riskmodel_dict.keys():
            raise ValueError('covariance panel missing from the risk model dictionary')
        if 'vol' not in riskmodel_dict.keys():
            raise ValueError('volatility data frame missing from the risk model dictionary')
        if not ret_df.index.equals(riskmodel_dict['cov'].items):
            raise ValueError('ret_df and risk model covariance panel should have the same date index')
        if not ret_df.columns.equals(riskmodel_dict['cov'].major_axis):
            raise ValueError('ret_df and risk model covariance panel should have the same assets')
        if not ret_df.index.equals(riskmodel_dict['vol'].index):
            raise ValueError('ret_df and risk model volatility data frame should have the same date index')
        if not ret_df.columns.equals(riskmodel_dict['vol'].columns):
            raise ValueError('ret_df and risk model volatility data frame should have the same assets')

    linear_tcosts_df = _reshape_tcosts(linear_tcosts, holdings_df)

    # backtest dictionary
    backtest_dict = {}
    backtest_dict['returns'] = ret_df
    if score_df is not None:
        backtest_dict['score'] = score_df
    if alpha_df is not None:
        backtest_dict['alpha'] = alpha_df

    backtest_dict['holdings'] = holdings_df
    backtest_dict['holdings_net'] = pd.DataFrame(holdings_df.sum(axis = 1), columns = ['Net Holdings'])
    backtest_dict['holdings_tilt'] = calc_tilt_holdings(holdings_df, tilt_period=tilt_period)
    backtest_dict['holdings_timing'] = backtest_dict['holdings'] - backtest_dict['holdings_tilt']

    va_df = calc_value_added(ret_df, holdings_df, linear_tcosts=linear_tcosts_df)
    cva_df = calc_value_added(ret_df, holdings_df, linear_tcosts=linear_tcosts_df, cumulate=True, cumulate_method=cumulate_method)
    portfolio_va_df = calc_value_added(ret_df, holdings_df, linear_tcosts=linear_tcosts_df, portfolio=True)
    portfolio_cva_df = calc_value_added(ret_df, holdings_df, linear_tcosts=linear_tcosts_df, cumulate=True, portfolio=True, cumulate_method=cumulate_method)

    backtest_dict['va'] = va_df
    backtest_dict['cva'] = cva_df
    backtest_dict['portfolio_va'] = portfolio_va_df
    backtest_dict['portfolio_cva'] = portfolio_cva_df

    backtest_dict['trailing_expost_risk'] = _calc_ewma_volatility(portfolio_va_df,vol_halflife=trailing_expost_period,vol_seed_period=trailing_expost_period,
                                                                     var_annualization_factor=annualization_factor,
                                                                     lag=0)
    if trailing_expost_period == 5:
        expost_tag = '1W'
    elif trailing_expost_period == 21:
        expost_tag = '1M'
    elif trailing_expost_period == 63:
        expost_tag = '3M'
    elif trailing_expost_period == 126:
        expost_tag = '6M'
    elif trailing_expost_period == 252:
        expost_tag = '1Y'
    else:
        expost_tag = str(trailing_expost_period) + 'D'
    backtest_dict['trailing_expost_risk'].columns = ['Portfolio Expost Risk (trailing ' + expost_tag + ')']

    tilt_va_df = calc_value_added(ret_df, backtest_dict['holdings_tilt'], linear_tcosts=linear_tcosts_df, portfolio=True, portfolio_name='Tilt')
    tilt_cva_df = calc_value_added(ret_df, backtest_dict['holdings_tilt'], linear_tcosts=linear_tcosts_df, portfolio=True, cumulate=True, cumulate_method=cumulate_method, portfolio_name='Tilt')
    timing_va_df = calc_value_added(ret_df, backtest_dict['holdings_timing'], linear_tcosts=linear_tcosts_df, portfolio=True, portfolio_name='Timing')
    timing_cva_df = calc_value_added(ret_df, backtest_dict['holdings_timing'], linear_tcosts=linear_tcosts_df, portfolio=True, cumulate=True, cumulate_method=cumulate_method, portfolio_name='Timing')

    backtest_dict['tilt_timing_va'] = portfolio_va_df.join(tilt_va_df).join(timing_va_df)
    backtest_dict['tilt_timing_cva'] = portfolio_cva_df.join(tilt_cva_df).join(timing_cva_df)

    if ret_df.shape[1] < number_of_assets_max_threshold:
        backtest_dict['IR_off_the_top'] = calc_off_the_top_IR(ret_df, holdings_df, linear_tcosts=linear_tcosts_df, annualization_factor=annualization_factor)
    backtest_dict['IR_lead_lag'] = calc_lead_lag_IR(ret_df, holdings_df, linear_tcosts=linear_tcosts_df, annualization_factor=annualization_factor, lead_lag_period=lead_lag_period)
    if alpha_df is not None:
        backtest_dict['IC_realized'], backtest_dict['IC_pval'] = calc_realized_IC(ret_df, alpha_df, riskmodel_dict['vol'])

    backtest_dict['performance_metrics'] = calc_portfolio_performance_metrics(portfolio_va_df, holdings_df, annualization_factor=annualization_factor)
    
    # portfolio VA by month and year
    backtest_dict['portfolio_va_by_month'] = calc_monthly_value_added(portfolio_va_df, cumulate_method=cumulate_method)
    
    # cva by longs and shorts      
    holdings_long_df = holdings_df.copy()
    holdings_long_df[holdings_long_df < 0] = 0
    holdings_short_df = holdings_df.copy()
    holdings_short_df[holdings_short_df > 0] = 0
    portfolio_long_va_df = calc_value_added(ret_df, holdings_long_df, linear_tcosts=linear_tcosts_df, portfolio=True, portfolio_name='Long')
    portfolio_long_cva_df = calc_value_added(ret_df, holdings_long_df, linear_tcosts=linear_tcosts_df, portfolio=True, cumulate=True, cumulate_method=cumulate_method, portfolio_name='Long')
    portfolio_short_va_df = calc_value_added(ret_df, holdings_short_df, linear_tcosts=linear_tcosts_df, portfolio=True, portfolio_name='Short')
    portfolio_short_cva_df = calc_value_added(ret_df, holdings_short_df, linear_tcosts=linear_tcosts_df, portfolio=True, cumulate=True, cumulate_method=cumulate_method, portfolio_name='Short')

    backtest_dict['portfolio_long_short_va'] = portfolio_va_df.join(portfolio_long_va_df).join(portfolio_short_va_df)
    backtest_dict['portfolio_long_short_cva'] = portfolio_cva_df.join(portfolio_long_cva_df).join(portfolio_short_cva_df)    
      
    # cva by cross-sectional and net positions
    if riskmodel_dict is not None and alpha_df is not None:
        if optimizer == 'mean_variance':
            holdings_crosssectional_df = mean_variance_optimizer(alpha_df,
                                                             riskmodel_dict['cov'],
                                                             risk_aversion=risk_aversion,
                                                             tcost_aversion=tcost_aversion,
                                                             linear_tcosts=linear_tcosts,
                                                             portfolio_lower_bound=-crosssectional_bound,
                                                             portfolio_upper_bound=crosssectional_bound)
            backtest_dict['holdings_crosssectional'] = holdings_crosssectional_df
        
            crosssectional_neutral_portfolio_va_df = calc_value_added(ret_df, holdings_crosssectional_df, linear_tcosts=linear_tcosts_df, portfolio=True)
            crosssectional_neutral_portfolio_cva_df = calc_value_added(ret_df, holdings_crosssectional_df, linear_tcosts=linear_tcosts_df, cumulate=True, portfolio=True, cumulate_method=cumulate_method)
            net_portfolio_va_df = portfolio_va_df - crosssectional_neutral_portfolio_va_df
            net_portfolio_cva_df = portfolio_cva_df - crosssectional_neutral_portfolio_cva_df
            crosssectional_neutral_portfolio_va_df.columns = ['Cross-sectional']
            crosssectional_neutral_portfolio_cva_df.columns = ['Cross-sectional']
            net_portfolio_va_df.columns = ['Net']
            net_portfolio_cva_df.columns = ['Net']
        
            backtest_dict['crosssectional_net_va'] = portfolio_va_df.join(crosssectional_neutral_portfolio_va_df).join(net_portfolio_va_df)
            backtest_dict['crosssectional_net_cva'] = portfolio_cva_df.join(crosssectional_neutral_portfolio_cva_df).join(net_portfolio_cva_df)
        else:
            raise ValueError('%s is not a valid optimizer' % optimizer)
    
    # IR by year
    backtest_dict['IR_by_year'] = calc_IR_by_year(portfolio_va_df)
    
    # rolling IR
    IR_rolling_dict = {}
    windows = {252: 'Rolling 1Y IR', 
               504: 'Rolling 2Y IR', 
               756: 'Rolling 3Y IR'}
    for window in windows.keys():
        IR_rolling_dict[windows[window]] = calc_IR_rolling(portfolio_va_df, annualization_factor=annualization_factor, window=window)
        IR_rolling_dict[windows[window]] = IR_rolling_dict[windows[window]].squeeze()
    backtest_dict['IR_rolling'] = pd.DataFrame(IR_rolling_dict)
    backtest_dict['IR_rolling_assets_1Y'] = calc_IR_rolling(va_df, annualization_factor=annualization_factor, window=252)
    
    # leverage
    backtest_dict['leverage'] = pd.DataFrame(holdings_df.abs().sum(axis=1, numeric_only=True).round(decimals=2), columns=['Leverage'])
    
    # rolling turnover
    abs_trades_df = backtest_dict['holdings'].fillna(0).diff().abs()
    backtest_dict['rolling_turnover'] = pd.rolling_sum(abs_trades_df, rolling_turnover_period).sum(axis=1) / pd.rolling_count(abs_trades_df, rolling_turnover_period).max(axis=1) * annualization_factor
    col_name = 'Rolling Turnover ' + str(rolling_turnover_period/252) + 'Y'
    backtest_dict['rolling_turnover'] = pd.DataFrame(backtest_dict['rolling_turnover'], columns = [col_name])
    
    # risk analysis
    if riskmodel_dict is not None:
        portfolio_exante_risk = {}
        portfolio_exante_variance = {}
        marginal_risk_contribution = {}
        risk_contribution = {}
        variance_contribution = {}

        orig_index = riskmodel_dict['cov'].items
        cov = riskmodel_dict['cov'].dropna(how='all')

        for idx, V in cov.iteritems():
            V_clean = V.dropna(axis=0, how='all').dropna(axis=1, how='all')
            if not V_clean.empty:
                current_holdings = holdings_df.ix[idx].dropna()
                active_assets = V_clean.index & current_holdings.index
                V_clean = pd.DataFrame(V_clean, index=active_assets, columns=active_assets)
                active_holdings = current_holdings[V_clean.index]

                V_dot_holdings = V_clean.dot(active_holdings)
                portfolio_exante_variance[idx] = active_holdings.dot(V_dot_holdings) * annualization_factor
                variance_contribution[idx] = active_holdings * V_dot_holdings * annualization_factor

                portfolio_exante_risk[idx] = math.sqrt(portfolio_exante_variance[idx])
                # avoid division by 0
                if portfolio_exante_risk[idx] != 0:
                    marginal_risk_contribution[idx] = (V_dot_holdings * annualization_factor) / portfolio_exante_risk[idx]
                    risk_contribution[idx] = active_holdings * marginal_risk_contribution[idx]

        portfolio_exante_variance_df = pd.DataFrame(portfolio_exante_variance, index=['Portfolio Exante Variance'], columns=orig_index).T.astype(np.float64)
        portfolio_exante_risk_df = pd.DataFrame(portfolio_exante_risk, index=['Portfolio Exante Risk'], columns=orig_index).T.astype(np.float64)
        marginal_risk_contribution_df = pd.DataFrame(marginal_risk_contribution, index=holdings_df.columns, columns=orig_index).T.astype(np.float64)
        risk_contribution_df = pd.DataFrame(risk_contribution, index=holdings_df.columns, columns=orig_index).T.astype(np.float64)
        variance_contribution_df = pd.DataFrame(variance_contribution, index=holdings_df.columns, columns=orig_index).T.astype(np.float64)
        risk_weight_df = variance_contribution_df.div(portfolio_exante_variance_df['Portfolio Exante Variance'], axis=0)

        backtest_dict['portfolio_exante_variance'] = portfolio_exante_variance_df
        backtest_dict['portfolio_exante_risk'] = portfolio_exante_risk_df
        backtest_dict['marginal_risk_contribution'] = marginal_risk_contribution_df
        backtest_dict['risk_contribution'] = risk_contribution_df
        backtest_dict['variance_contribution'] = variance_contribution_df
        backtest_dict['risk_weight'] = risk_weight_df
    
    # Value-at-Risk
    backtest_dict['VaR'] = calc_HS_VaR(portfolio_va_df).join(calc_Gaussian_VaR(portfolio_va_df))

    # drawdown analysis
    backtest_dict['drawdown'] = calc_drawdown(portfolio_va_df)
    backtest_dict['drawdown_periods'] = calc_drawdown_periods(backtest_dict['drawdown'])    
    
    # add parameters
    params = {}
    params['risk_aversion'] = risk_aversion
    params['tcost_aversion'] = tcost_aversion
    params['annualization_factor'] = annualization_factor
    params['tilt_period'] = tilt_period
    params['lead_lag_period'] = lead_lag_period
    params['cumulate_method'] = cumulate_method
    backtest_dict['params'] = params
    
    return backtest_dict

def backtest_risk_metrics_for_axioma_riskmodel(holdings_df,
                                               cov_files_on_disk,
                                               axiomaid_bloombergid_mapping_dict,
                                               annualization_factor=252):
    """
    Calculate backtest risk metrics, given an axioma risk model on disk

    Parameters
    ----------
    holdings_df : DataFrame
        DataFrame containing signal holdings for assets
    cov_files_on_disk : list
        List containing the axioma risk model (covariance panel) file names on disk
    axiomaid_bloombergid_mapping_dict : dict
        Dictionary containing the mapping from axioma ids to composite bloomberg ids
    annualization_factor : int, optional
        Factor used to annualize IR (defaults to 252, assuming daily data and 252 days (periods) to calculate risk and returns)

    Returns
    -------
    riskmetrics_dict : dict
        Dictionary object containing risk metrics

    """

    assets = holdings_df.columns
    periods = holdings_df.index

    portfolio_exante_risk = {}
    portfolio_exante_variance = {}
    marginal_risk_contribution = {}
    risk_contribution = {}
    variance_contribution = {}

    for cov_file in cov_files_on_disk:
        print('Processing : ' + cov_file)
        with pd.HDFStore(cov_file) as cov_link:
            cov = cov_link['CovPanel'].rename(major_axis=axiomaid_bloombergid_mapping_dict,
                                              minor_axis=axiomaid_bloombergid_mapping_dict)
        intersection_assets = assets.intersection(cov.major_axis)
        cov = cov.reindex(major_axis=intersection_assets, minor_axis=intersection_assets).dropna(how='all').div(annualization_factor)

        for idx, V in cov.iteritems():
            V_clean = V.dropna(axis=0, how='all').dropna(axis=1, how='all')
            if not V_clean.empty and idx in holdings_df.index:
                current_holdings = holdings_df.ix[idx].dropna()
                active_assets = V_clean.index & current_holdings.index
                V_clean = pd.DataFrame(V_clean, index=active_assets, columns=active_assets)
                active_holdings = current_holdings[V_clean.index]

                V_dot_holdings = V_clean.dot(active_holdings)
                portfolio_exante_variance[idx] = active_holdings.dot(V_dot_holdings) * annualization_factor
                variance_contribution[idx] = active_holdings * V_dot_holdings * annualization_factor

                portfolio_exante_risk[idx] = math.sqrt(portfolio_exante_variance[idx])
                # avoid division by 0
                if portfolio_exante_risk[idx] != 0:
                    marginal_risk_contribution[idx] = (V_dot_holdings * annualization_factor) / portfolio_exante_risk[idx]
                    risk_contribution[idx] = active_holdings * marginal_risk_contribution[idx]

    riskmetrics_dict = {}
    riskmetrics_dict['portfolio_exante_variance'] = pd.DataFrame(portfolio_exante_variance, index=['Portfolio Exante Variance'],columns=periods).T.astype(np.float64)
    riskmetrics_dict['portfolio_exante_risk'] = pd.DataFrame(portfolio_exante_risk, index=['Portfolio Exante Risk'],columns=periods).T.astype(np.float64)
    riskmetrics_dict['marginal_risk_contribution'] = pd.DataFrame(marginal_risk_contribution, index=holdings_df.columns,columns=periods).T.astype(np.float64)
    riskmetrics_dict['risk_contribution'] = pd.DataFrame(risk_contribution, index=holdings_df.columns, columns=periods).T.astype(np.float64)
    riskmetrics_dict['variance_contribution'] = pd.DataFrame(variance_contribution, index=holdings_df.columns, columns=periods).T.astype(np.float64)
    riskmetrics_dict['risk_weight'] = riskmetrics_dict['variance_contribution'].div(riskmetrics_dict['portfolio_exante_variance']['Portfolio Exante Variance'], axis=0)

    return riskmetrics_dict

def seasonality_metrics(va_df):
    """
    Calculate seasonality for different time horizons

    Parameters
    ----------
    va_df : DataFrame
        DataFrame containing value-added timeseries data for portfolio

    Returns
    -------
    seasonality_dict : dict
        Dictionary object containing seasonality metrics

    """
    if not isinstance(va_df, pd.DataFrame):
        raise TypeError('va_df should be a DataFrame object')
        
    # select indices based on frequency
    df = pd.DataFrame(list(map(lambda x: 'Q%d' % x.quarter, va_df.index)), 
                      index=va_df.index,
                      columns=['quarter'])
    df['month'] = list(map(lambda x: calendar.month_name[x.month][:3], va_df.index))
    df['day_of_week'] = list(map(lambda x: x.isoweekday(), va_df.index))
    df['business_day_of_month'] = 'Others'
    gby = df.groupby(lambda x: (x.year, x.month))
    df.ix[gby.nth(0).index,'business_day_of_month'] = 'First'
    df.ix[gby.nth(1).index,'business_day_of_month'] = 'Second'
    df.ix[gby.nth(2).index,'business_day_of_month'] = 'Third'
    df.ix[gby.nth(3).index,'business_day_of_month'] = 'Fourth'
    df.ix[gby.nth(4).index,'business_day_of_month'] = 'Fifth'
    df.ix[gby.nth(-1).index,'business_day_of_month'] = 'Last'
    df.ix[gby.nth(-2).index,'business_day_of_month'] = 'Last-1'
    df.ix[gby.nth(-3).index,'business_day_of_month'] = 'Last-2'
    df.ix[gby.nth(-4).index,'business_day_of_month'] = 'Last-3'
    df.ix[gby.nth(-5).index,'business_day_of_month'] = 'Last-4'
    
    seasonality_dict = {}
    
    # seasonality by month
    freq = df['month']
    grouped = va_df.groupby(freq)
    metrics = {}
    for name, group in grouped:
        metrics[name] = _calc_return_stats(group.squeeze()) 
    ordering = [calendar.month_name[x][:3] for x in range(1, 13)]
    seasonality_dict['month'] = pd.DataFrame(metrics).T.ix[ordering]
   
    # seasonality by quarter
    freq = df['quarter']
    grouped = va_df.groupby(freq)
    metrics = {}
    for name, group in grouped:
        metrics[name] = _calc_return_stats(group.squeeze()) 
    seasonality_dict['quarter'] = pd.DataFrame(metrics).T
    
    # seasonality by day of week
    freq = df['day_of_week']
    weekday = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    freq = freq.map(lambda x: weekday[x - 1])
    grouped = va_df.groupby(freq)
    metrics = {}
    for name, group in grouped:
        metrics[name] = _calc_return_stats(group.squeeze())
    seasonality_dict['day_of_week'] = pd.DataFrame(metrics).T.ix[weekday]
    
    # seasonality by business day of month
    freq = df['business_day_of_month']
    grouped = va_df.groupby(freq)
    metrics = {}
    for name, group in grouped:
        metrics[name] = _calc_return_stats(group.squeeze()) 
    ordering = ['First','Second','Third','Fourth','Fifth','Last-4','Last-3','Last-2','Last-1','Last']
    seasonality_dict['business_day_of_month'] = pd.DataFrame(metrics).T.ix[ordering]
    
    return seasonality_dict
