# Copyright (C) 2015 State Street Global Advisors

import pandas as pd
import numpy as np
import cvxopt as copt


def _validate_bound(bound, assets, constraint_name, periods=None):
    if pd.isnull(bound.values).any():
        raise ValueError(constraint_name + 'can not have nan values')
    elif not bound.axes[-1].equals(assets):
        raise ValueError(constraint_name + ' does not have values for all assets')
    elif len(bound.axes) == 3:
        assert periods is not None
        if not bound.axes[0].equals(periods):
            # Only check this for panels
            raise ValueError(constraint_name + ' does not have values for all periods in the index')
    return bound


def _bound_weights_to_current_constraint(bound_weights, idx, multiple_portfolio_bounds):
    if isinstance(bound_weights, pd.Panel):
        pbound_weights = bound_weights.ix[idx]
    elif isinstance(bound_weights, pd.DataFrame):
        if multiple_portfolio_bounds:
            pbound_weights = bound_weights
        else:
            pbound_weights = bound_weights.ix[idx]
    else:
        pbound_weights = bound_weights

    return pbound_weights


def _convert_to_bound(bound, alpha_df, name):
    if isinstance(bound, pd.Series):
        bound = _validate_bound(bound, alpha_df.columns, name)
    elif not isinstance(bound, float):
        raise TypeError(name + ' should be a float or Series object')
    else:
        bound = pd.Series(bound, index=alpha_df.columns)

    return bound


def _validate_portfolio_bound(portfolio_bound, bound_weights, constraint_name):
    if isinstance(bound_weights, pd.Series):
        raise ValueError('The number of constraints in bound weights does not match the number of bounds specified in ' + constraint_name)

    if isinstance(portfolio_bound, pd.Series):
        if isinstance(portfolio_bound.index, pd.DatetimeIndex):
            if isinstance(bound_weights, pd.DataFrame):
                if isinstance(bound_weights.index, pd.DatetimeIndex):
                    if not portfolio_bound.index.equals(bound_weights.index):
                        raise ValueError('The index of bound weights does not match the index of ' + constraint_name)
                elif portfolio_bound.shape[0] > 1:
                    raise ValueError('The number of constraints in bound weights does not match the number of bounds specified in ' + constraint_name)
            elif isinstance(bound_weights, pd.Panel):
                if not portfolio_bound.index.equals(bound_weights.items):
                    raise ValueError('The items (time index) of bound weights does not match the index of ' + constraint_name)
                elif bound_weights.shape[1] > 1:
                    raise ValueError('The number of constraints in bound weights does not match the number of bounds specified in ' + constraint_name)
        else:
            if isinstance(bound_weights, pd.DataFrame) and not isinstance(bound_weights.index,pd.DatetimeIndex) and portfolio_bound.shape[0] != bound_weights.shape[0]:
                raise ValueError('The number of constraints in bound weights does not match the number of bounds specified in ' + constraint_name)
            elif isinstance(bound_weights, pd.Panel) and portfolio_bound.shape[0] != bound_weights.shape[1]:
                raise ValueError('The number of constraints in bound weights does not match the number of bounds specified in ' + constraint_name)
    elif isinstance(portfolio_bound, pd.DataFrame):
        if isinstance(bound_weights, pd.DataFrame) and not portfolio_bound.index.equals(bound_weights.index):
            raise ValueError('The index of bound weights does not match the index of ' + constraint_name)
        elif isinstance(bound_weights, pd.DataFrame) and portfolio_bound.shape[1] != 1:
            raise ValueError('The number of constraints in bound weights does not match the number of bounds specified in ' + constraint_name)
        elif isinstance(bound_weights, pd.Panel) and not portfolio_bound.index.equals(bound_weights.items):
            raise ValueError('The index of bound weights does not match the index of ' + constraint_name)
        elif isinstance(bound_weights, pd.Panel) and portfolio_bound.shape[1] != bound_weights.shape[1]:
            raise ValueError('The number of constraints in bound weights does not match the number of bounds specified in ' + constraint_name)


def _add_portfolio_constraints(G, h, portfolio_bound_weights, portfolio_lower_bound, portfolio_upper_bound, valid_assets, initial_holdings, dimension_multiplier):
    """
    Update optimizer parameters (G and h) to reflect provided portfolio constraints (portfolio bound weights, portfolio lower and upper bounds)
    """
    if portfolio_bound_weights is not None:
        if isinstance(portfolio_bound_weights, pd.Series):
            portfolio_bound_weights = portfolio_bound_weights[valid_assets].values
            weighted_initial_holdings = portfolio_bound_weights.dot(initial_holdings)
        elif isinstance(portfolio_bound_weights, pd.DataFrame):
            if portfolio_bound_weights.shape[0] > 1:
                portfolio_bound_weights = portfolio_bound_weights[valid_assets].values
            else:
                portfolio_bound_weights = portfolio_bound_weights[valid_assets].values.flatten()
            weighted_initial_holdings = portfolio_bound_weights.dot(initial_holdings)
    else:
        portfolio_bound_weights = np.ones(len(valid_assets))
        weighted_initial_holdings = portfolio_bound_weights.dot(initial_holdings)

    has_lower_bound = False
    if portfolio_lower_bound is not None:
        if isinstance(portfolio_lower_bound, pd.Series):
            portfolio_lower_bound = portfolio_lower_bound.values
        elif not isinstance(portfolio_lower_bound, float):
            raise TypeError('portfolio_lower_bound should be a float or Series object')
        has_lower_bound = True
        new_portfolio_lower_bound = portfolio_lower_bound - weighted_initial_holdings
        if dimension_multiplier == 1:
            G = np.vstack((G, -portfolio_bound_weights))
            h = np.hstack((h, -new_portfolio_lower_bound))
        elif dimension_multiplier == 2:
            G = np.vstack((G, np.hstack((portfolio_bound_weights, -portfolio_bound_weights))))
            h = np.hstack((h, new_portfolio_lower_bound))

    has_upper_bound = False
    if portfolio_upper_bound is not None:
        if isinstance(portfolio_upper_bound, pd.Series):
            portfolio_upper_bound = portfolio_upper_bound.values
        elif not isinstance(portfolio_upper_bound, float):
            raise TypeError('portfolio_upper_bound should be a float or Series object')
        has_upper_bound = True
        new_portfolio_upper_bound = portfolio_upper_bound - weighted_initial_holdings
        if dimension_multiplier == 1:
            G = np.vstack((G, portfolio_bound_weights))
            h = np.hstack((h, new_portfolio_upper_bound))
        elif dimension_multiplier == 2:
            G = np.vstack((G, np.hstack((portfolio_bound_weights, -portfolio_bound_weights))))
            h = np.hstack((h, new_portfolio_upper_bound))

    # if only upper bound or lower bound is given while the other is missing, to stabilize the
    # optimizer artificially add an unattainable lower or upper bound to the constraints
    artificial_large_upper_bound = 100.0
    artificial_large_lower_bound = -100.0
    if has_upper_bound and not has_lower_bound:
        if isinstance(portfolio_upper_bound, float):
            if dimension_multiplier == 1:
                G = np.vstack((G, -portfolio_bound_weights))
                h = np.hstack((h, -artificial_large_lower_bound))
    if has_lower_bound and not has_upper_bound:
        if isinstance(portfolio_lower_bound, float):
            if dimension_multiplier == 1:
                G = np.vstack((G, portfolio_bound_weights))
                h = np.hstack((h, artificial_large_upper_bound))

    return G, h


def mean_variance_optimizer(alpha_df,
                            cov_panel,
                            risk_aversion=1.0,
                            tcost_aversion=1.0,
                            linear_tcosts=None,
                            position_lower_bound=None,
                            position_upper_bound=None,
                            portfolio_bound_weights=None,
                            portfolio_lower_bound=None,
                            portfolio_upper_bound=None):
    """
    Mean variance optimizer with transaction costs

    Parameters
    ----------
    alpha_df : DataFrame
        DataFrame object containing timeseries of alphas (expected returns) for assets
    cov_panel : Panel
        Panel object containing asset covariance matrix
    risk_aversion : float, optional
        Risk aversion parameter is used to modulate portfolio's ex-post risk
    tcost_aversion : float, optional
        Transaction-cost aversion parameter is used to modulate aversion to transaction costs in the optimizer
    linear_tcosts : float, Series or DataFrame, optional
        Linear transaction costs are applicable for unit change in asset holdings
        If linear_tcosts is float, the same linear transaction costs are applied to all assets
        If linear_tcosts is a Series object, the vector specifies the transaction costs applicable for each asset
        If linear_tcosts is a DataFrame object, then the DataFrame specifies transaction costs applicable for each asset (column) for each period (index)
    position_lower_bound : float or Series, optional
        Position lower bound parameter is used to apply a lower bound on asset positions
        If position_lower_bound is float, the same lower bound is applied on all asset positions
        If position_lower_bound is a Series object, the vector specifies asset specific lower bounds
    position_upper_bound : float or Series, optional
        Position upper bound parameter is used to apply a upper bound on asset positions
        If position_upper_bound is float, the same upper bound is applied on all asset positions
        If position_upper_bound is a Series object, the vector specifies asset specific upper bounds
    portfolio_bound_weights : Series or DataFrame or Panel, optional
        This parameter specifies the weights applied to portfolio holdings to constrain the aggregate holdings as
        specified by the portfolio_lower_bound and portfolio_upper_bound parameters.
        By default, when this parameter is not specified or is None, the weights are assumed to be a
        vector of 1s, i.e. a straight sum of portfolio holdings.
        If the weights are specified as a Series, the weights will be applied to constrain holdings each day.
        If the weights are specified as a DataFrame, each row specifies a single constraint applied to holdings each day.
        The number of rows in the DataFrame needs to be the same as the size of the portfolio upper/lower bound values.
        If the weights are specified as a Panel, the axes should be ordered as [time, constraint, asset] and
        the hence the portfolio bound weights can be time varying.
    portfolio_lower_bound : float or Series or DataFrame, optional
        This parameter specifies the lower bound for portfolio aggregate holdings (defaults to None)
        If the portfolio_bound_weights is a Series, then the portfolio_lower_bound should be specified as a float i.e. one constraint only.
        If the portfolio_bound_weights is a DataFrame, then the portfolio_lower_bound should be specified either as a Series or DataFrame
        where each element in the Series or DataFrame applies to a correponding row in portfolio_bound_weights.
        If the portfolio_bound_weights is a Panel, then the portfolio_lower_bound should be specified either as a Series
        or DataFrame. If it is a Series, then each element in the Series applies to a correponding constraint in
        portfolio_bound_weights through time. If it is a DataFrame, than each row in the DataFrame applies to the corresponding
        constraints in portfolio_bound_weights through time.
        To apply one-sided constraints, the portfolio_lower_bound can be either set to NaN or None as the bound value.
    portfolio_upper_bound : float or Series or DataFrame, optional
        This parameter specifies the upper bound for portfolio aggregate holdings (defaults to None)
        If the portfolio_bound_weights is a Series, then the portfolio_upper_bound should be specified as a float i.e. one constraint only.
        If the portfolio_bound_weights is a DataFrame, then the portfolio_upper_bound should be specified either as a Series or DataFrame
        where each element in the Series or DataFrame applies to a correponding row in portfolio_bound_weights.
        If the portfolio_bound_weights is a Panel, then the portfolio_lower_bound should be specified either as a Series
        or DataFrame. If it is a Series, then each element in the Series applies to a correponding constraint in
        portfolio_bound_weights through time. If it is a DataFrame, than each row in the DataFrame applies to the corresponding
        constraints in portfolio_bound_weights through time.
        To apply one-sided constraints, the portfolio_lower_bound can be either set to NaN or None as the bound value.

    Returns
    -------
    holdings_df : DataFrame
        DataFrame object containing timeseries of optimal holdings generated by the mean-variance optimizer

    Notes
    -----
    The objective function for the optimizer is as follows:

    max_h : alpha' * h - lambda * h' * V * h - k * (a + b * [(PV*|h - hi|)/BT]^c)|h - hi|

    where

        alpha : expected returns
        h     : optimal asset holdings
        hi    : initial asset holdings
        lambda: risk aversion parameter
        k     : transaction cost aversion parameter
        a     : linear transaction costs
        b     : market impact transaction costs
        PV    : portfolio value
        BT    : largest trade or average daily volume
        c     : transaction cost exponent

    """

    if not isinstance(alpha_df, pd.DataFrame):
        raise TypeError('alphas_df should be a DataFrame object')
    if not isinstance(cov_panel, pd.Panel):
        raise TypeError('cov_panel should be a Panel object')
    if risk_aversion <= 0:
        raise ValueError('%f is not a valid risk aversion' % risk_aversion)
    if tcost_aversion <= 0:
        raise ValueError('%f is not a valid tcost aversion' % tcost_aversion)

    if not alpha_df.index.equals(cov_panel.items):
        raise ValueError('alphas_df and cov_panel should have the same date index')
    if not alpha_df.columns.equals(cov_panel.major_axis):
        raise ValueError('alphas_df and cov_panel should have the same assets')
    if not cov_panel.major_axis.equals(cov_panel.minor_axis):
        raise ValueError('cov_panel is not a valid covariance matrix')

    if linear_tcosts is not None:
        if isinstance(linear_tcosts, pd.DataFrame):
            if linear_tcosts.isnull().any().any():
                raise ValueError('linear_tcosts can not have nan transaction costs')
            elif not linear_tcosts.index.equals(alpha_df.index):
                raise ValueError('linear_tcosts and alphas_df should have the same date index')
            elif not linear_tcosts.columns.equals(alpha_df.columns):
                raise ValueError('linear_tcosts and alphas_df should have the same assets')
            else:
                linear_tcosts_df = linear_tcosts.reindex(columns=alpha_df.columns)
        elif isinstance(linear_tcosts, pd.Series):
            if linear_tcosts.isnull().any():
                raise ValueError('linear_tcosts can not have nan transaction costs')
            elif not linear_tcosts.index.equals(alpha_df.columns):
                raise ValueError('linear_tcosts does not have transaction costs for all assets')
            else:
                linear_tcosts = linear_tcosts.reindex(index=alpha_df.columns)
                linear_tcosts_df = pd.DataFrame(np.nan, index=alpha_df.index, columns=alpha_df.columns)
                linear_tcosts_df.ix[0] = linear_tcosts
                linear_tcosts_df = linear_tcosts_df.ffill()
        elif not isinstance(linear_tcosts, float):
            raise TypeError('linear_tcosts should be a float, Series or DataFrame object')
        else:
            linear_tcosts_df = pd.DataFrame(linear_tcosts, index=alpha_df.index, columns=alpha_df.columns)
    else:
        linear_tcosts_df = pd.DataFrame(0.0, index=alpha_df.index, columns=alpha_df.columns)

    if position_lower_bound is not None:
        position_lower_bound = _convert_to_bound(position_lower_bound, alpha_df, 'position_lower_bound')

    if position_upper_bound is not None:
        position_upper_bound = _convert_to_bound(position_upper_bound, alpha_df, 'position_upper_bound')

    multiple_portfolio_bounds = False
    if portfolio_bound_weights is not None:
        if isinstance(portfolio_bound_weights, (pd.Series, pd.DataFrame)):
            portfolio_bound_weights = _validate_bound(portfolio_bound_weights, alpha_df.columns, 'portfolio_bound_weights')
            if isinstance(portfolio_bound_weights, pd.DataFrame) and not isinstance(portfolio_bound_weights.index, pd.DatetimeIndex):
                multiple_portfolio_bounds = True
        elif isinstance(portfolio_bound_weights, pd.Panel):
            portfolio_bound_weights = _validate_bound(portfolio_bound_weights, alpha_df.columns, 'portfolio_bound_weights', alpha_df.index)
            if portfolio_bound_weights.shape[1] > 1:
                multiple_portfolio_bounds = True
        else:
            raise TypeError('portfolio_bound_weights should be a Series/DataFrame/Panel object')

    port_bounds = [(portfolio_lower_bound, 'portfolio_lower_bound'), (portfolio_upper_bound, 'portfolio_upper_bound')]
    for port_val, port_name in port_bounds:
        if port_val is not None:
            if isinstance(port_val, (pd.Series, pd.DataFrame)):
                _validate_portfolio_bound(port_val, portfolio_bound_weights, port_name)
            elif isinstance(port_val, float):
                if isinstance(portfolio_bound_weights, pd.Panel) and portfolio_bound_weights.shape[1] != 1:
                    raise TypeError('The number of constraints in portfolio_bound_weights does not match the number of bounds specified in ' + port_name)
            else:
                raise TypeError(port_name + ' should be a float or Series or DataFrame object')

    holdings_dict = {}
    asset_universe = cov_panel.major_axis
    holdings_initial = pd.Series(np.nan, index=asset_universe)

    for idx in cov_panel.items:
        cov = cov_panel.ix[idx]
        alpha = alpha_df.ix[idx]
        tcosts = linear_tcosts_df.ix[idx]

        pbound_weights = _bound_weights_to_current_constraint(portfolio_bound_weights, idx, multiple_portfolio_bounds)
        pbound_lower = _bound_weights_to_current_constraint(portfolio_lower_bound, idx, multiple_portfolio_bounds)
        pbound_upper = _bound_weights_to_current_constraint(portfolio_upper_bound, idx, multiple_portfolio_bounds)

        holdings_dict[idx] = mean_variance_optimizer_one_period(alpha, cov,
                                                                risk_aversion=risk_aversion,
                                                                tcost_aversion=tcost_aversion,
                                                                linear_tcosts=tcosts,
                                                                position_lower_bound=position_lower_bound,
                                                                position_upper_bound=position_upper_bound,
                                                                initial_holdings=holdings_initial,
                                                                portfolio_bound_weights=pbound_weights,
                                                                portfolio_lower_bound=pbound_lower,
                                                                portfolio_upper_bound=pbound_upper)
        holdings_initial = holdings_dict[idx]

    # optimal holdings
    holdings_df = pd.DataFrame.from_dict(holdings_dict, orient='index', dtype=float)

    return holdings_df


def mean_variance_optimizer_one_period(alpha, cov_df, risk_aversion=1.0, tcost_aversion=1.0, linear_tcosts=None,
                                       position_lower_bound=None, position_upper_bound=None, initial_holdings=None,
                                       portfolio_bound_weights=None, portfolio_lower_bound=None, portfolio_upper_bound=None):
    """
    Mean variance optimizer with transaction costs

    Parameters
    ----------
    alpha : Series
        Series object containing alphas (expected returns) for assets in the current period
    cov_df : DataFrame
        DataFrame object containing asset covariance matrix for current period
    risk_aversion : float, optional
        Risk aversion parameter is used to modulate portfolio's ex-post risk
    tcost_aversion : float, optional
        Transaction-cost aversion parameter is used to modulate aversion to transaction costs in the optimizer
    linear_tcosts : float, Series, optional
        Linear transaction costs are applicable for unit change in asset holdings
        If linear_tcosts is float, the same linear transaction costs are applied to all assets
        If linear_tcosts is a Series object, the vector specifies the transaction costs applicable for each asset
    position_lower_bound : float or Series, optional
        Position lower bound parameter is used to apply a lower bound on asset positions
        If position_lower_bound is float, the same lower bound is applied on all asset positions
        If position_lower_bound is a Series object, the vector specifies asset specific lower bounds
    position_upper_bound : float or Series, optional
        Position upper bound parameter is used to apply a upper bound on asset positions
        If position_upper_bound is float, the same upper bound is applied on all asset positions
        If position_upper_bound is a Series object, the vector specifies asset specific upper bounds
    initial_holdings : Series, optional
        Holdings from the prior period, only relevant if transaction costs are non-zero
    portfolio_bound_weights : Series or DataFrame, optional
        This parameter specifies the weights applied to portfolio holdings to constrain the aggregate holdings as
        specified by the portfolio_lower_bound and portfolio_upper_bound parameters.
        By default, when this parameter is not specified or is None, the weights are assumed to be a
        vector of 1s, i.e. a straight sum of portfolio holdings.
        If the weights are specified as a Series, the weights will be applied to constrain holdings each day.
        If the weights are specified as a DataFrame, each row specifies a single constraint applied to holdings each day.
        The number of rows in the DataFrame needs to be the same as the size of the portfolio upper/lower bound values.
    portfolio_lower_bound : float or Series, optional
        This parameter specifies the lower bound for portfolio aggregate holdings (defaults to None)
        If the portfolio_bound_weights is a Series, then the portfolio_lower_bound should be specified as a float i.e. one constraint only.
        If the portfolio_bound_weights is a DataFrame, then the portfolio_lower_bound should be specified as a Series
        where each element in the Series applies to a correponding row in portfolio_bound_weights.
        To apply one-sided constraints, the portfolio_lower_bound can be either set to NaN or None as the bound value.
    portfolio_upper_bound : float or Series, optional
        This parameter specifies the upper bound for portfolio aggregate holdings (defaults to None)
        If the portfolio_bound_weights is a Series, then the portfolio_upper_bound should be specified as a float i.e. one constraint only.
        If the portfolio_bound_weights is a DataFrame, then the portfolio_upper_bound should be specified as a Series
        where each element in the Series applies to a correponding row in portfolio_bound_weights.
        To apply one-sided constraints, the portfolio_lower_bound can be either set to NaN or None as the bound value.

    Returns
    -------
    holdings : Series
        Series object containing optimal holdings generated by the mean-variance optimizer

    Notes
    -----
    The objective function for the optimizer is as follows:

    max_h : alpha' * h - lambda * h' * V * h - k * (a + b * [(PV*|h - hi|)/BT]^c)|h - hi|

    where

        alpha : expected returns
        h     : optimal asset holdings
        hi    : initial asset holdings
        lambda: risk aversion parameter
        k     : transaction cost aversion parameter
        a     : linear transaction costs
        b     : market impact transaction costs
        PV    : portfolio value
        BT    : largest trade or average daily volume
        c     : transaction cost exponent

    """
    asset_universe = cov_df.columns
    cov_clean = cov_df.dropna(axis=0, how='all').dropna(axis=1, how='all')
    alpha_clean = alpha.dropna()

    if not cov_clean.index.equals(cov_clean.columns):
        raise ValueError("cleaned cov's index != columns: {}, {}".format(cov_clean.index, cov_clean.columns))

    if alpha_clean.empty or cov_clean.empty:
        # optimal holdings will be empty
        holdings = pd.Series(np.nan, index=asset_universe)
    else:
        valid_assets = cov_clean.index & alpha_clean.index
        cov_clean = cov_clean.ix[valid_assets, valid_assets]
        alpha_clean = alpha_clean[valid_assets]
        if linear_tcosts is not None:
            linear_tcosts = linear_tcosts.reindex(index=valid_assets)
            initial_holdings = initial_holdings.reindex(index=valid_assets)
            have_tcosts = True
        else:
            have_tcosts = False


        # set variables to capture linear t-costs and market impact t-costs
        # k1 = k * a, where k = t-cost aversion and a = linear t-costs
        if have_tcosts:
            k1 = tcost_aversion * np.array(linear_tcosts)
        else:
            k1 = None
        # k2 = k * b * [(PV*|h-hi|)/BT]^c, where k = t-cost aversion, b = market impact t-cost, PV = portfolio value,
        # BT = Largest trade or ADV (Average Daily Volume), c = t-cost exponent
        # for now, k2 is assumed to be 0
        # k2 = 0

        # we will now fit the problem into the standard qp form that's handled by CVXOPT qp function (given below)
        #
        # min_x  (1/2) * x' * P * x + q' * x
        #
        # subject to constraints :   G * x <= h (inequality constraint)
        #                            A * x = b  (equality constraint)
        #
        # This objective function is convex if and only if P is positive semi-definite

        if not have_tcosts or np.all(k1 == 0.0):
            P = 2 * risk_aversion * np.array(cov_clean)
            q = - np.array(alpha_clean)
            G = np.empty((0, len(valid_assets)))
            h = np.empty((0,))
            h_init = np.zeros(len(valid_assets))
            dimension_multiplier = 1
        else:
            #h_init = np.array(initial_holdings)
            #G = -1 * np.eye(2*len(valid_assets))
            #h = np.zeros(2*len(valid_assets))
            #dimension_multiplier = 2
            raise NotImplementedError("The current version of the optimizer does not handle transaction costs")

        if isinstance(position_lower_bound, float):
            position_lower_bound = pd.Series(position_lower_bound, index=valid_assets)

        if isinstance(position_upper_bound, float):
            position_upper_bound = pd.Series(position_upper_bound, index=valid_assets)

        if position_lower_bound is not None or position_upper_bound is not None:
            if dimension_multiplier == 1:
                G1 = np.eye(len(valid_assets))
            #elif dimension_multiplier == 2:
            #    G1 = np.hstack((np.eye(len(valid_assets)), -np.eye(len(valid_assets))))

        if position_upper_bound is not None:
            h1 = np.array(position_upper_bound.reindex(valid_assets)) - h_init
            G = np.vstack((G, G1))
            h = np.hstack((h, h1))

        if position_lower_bound is not None:
            h1 = np.array(position_lower_bound.reindex(valid_assets)) - h_init
            G = np.vstack((G, -G1))
            h = np.hstack((h, -h1))

        G, h = _add_portfolio_constraints(G, h, portfolio_bound_weights, portfolio_lower_bound, portfolio_upper_bound, valid_assets, h_init, dimension_multiplier)

        # call QP solver
        copt.solvers.options['show_progress'] = False
        copt_dict = copt.solvers.qp(P=copt.matrix(P), q=copt.matrix(q), G=copt.matrix(G), h=copt.matrix(h))

        # parse result from the solver
        if copt_dict['status'] == 'optimal':
            optimal_holdings = np.reshape(np.array(copt_dict['x']), (len(valid_assets),))
            holdings = pd.Series(optimal_holdings, index=valid_assets).reindex(asset_universe)
        else:
            raise NotImplementedError('cvxopt failed to solve the QP; handling this case is not yet implemented')

    return holdings
