

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
from typing import Union


def compute_non_iid_adjustment(realized_return: pd.Series,
                               q : int=252)-> float:

    """
    Calculate the adjustment factor for the Sharpe ratio when returns are not independent and identically distributed (non-iid).

    Parameters:
    ----------
    realized_return : pd.Series
        A Series representing the daily returns of the strategy.

    Returns:
    -------
    float
        The adjustment factor for the Sharpe ratio when returns are not iid.

    """
    eta = 0
    for k in range(1,q-1):
        rho = realized_return.autocorr(lag=k)
        eta = eta + (q-k)*rho
    compute_non_iid_adjustment = q/np.sqrt(q+2*eta)
    return compute_non_iid_adjustment




def sharpe_ratio(realized_return: pd.Series,
                 annualize: bool = True,
                 adjust_non_iid: bool = True) -> Union[float, float]:

    """
    Calculate the Sharpe ratio of a strategy.

    Parameters:
    ----------
    realized_return : pd.Series
        A Series representing the daily returns of the strategy.

    Returns:
    -------
    float
        The Sharpe ratio of the strategy.

    Notes:
    -----
    The Sharpe ratio is calculated as the average daily return divided by the standard deviation of the daily returns.
    The result is then annualized by multiplying by the square root of 252 (assuming 252 trading days in a year).
    """
    # Calculate the Sharpe ratio
    daily_sharpe_ratio = realized_return.mean() / realized_return.std()
    if annualize:
        if adjust_non_iid:
            compute_non_iid_adjustment = compute_non_iid_adjustment(realized_return)
            sharpe_ratio = daily_sharpe_ratio * compute_non_iid_adjustment
        else:
            sharpe_ratio = daily_sharpe_ratio * np.sqrt(252)
    else:
        sharpe_ratio = daily_sharpe_ratio


    number_of_observations = realized_return.shape[0]

    sr_standard_error = np.sqrt( ( 1 + 0.5 * (sharpe_ratio **2 )  ) / number_of_observations)
    return sharpe_ratio, sr_standard_error




def portfolio_volatility(weights: pd.Series,
                         covariance: Union[np.ndarray, pd.DataFrame]) -> float:
    """
    Calculate the annualized portfolio volatility.

    Parameters:
    ----------
    weights : pd.Series
        A Series representing the weights of the assets in the portfolio.
    covariance : np.ndarray or pd.DataFrame
        Either a NumPy array or a Pandas DataFrame representing the covariance matrix
        of the asset returns.

    Returns:
    -------
    float
        The annualized portfolio volatility.

    Notes:
    -----
    The function calculates the portfolio volatility using the weights and the covariance matrix.
    It then annualizes the volatility by multiplying by the square root of 252 (assuming 252 trading days in a year).
    """
    if isinstance(covariance, pd.DataFrame):
        covariance = covariance.values  # Convert DataFrame to NumPy array if it's a DataFrame

    # Calculate the portfolio variance
    portfolio_variance = np.dot(np.dot(weights.values, covariance), weights.values)

    # Calculate the annualized portfolio volatility
    annualized_volatility = np.sqrt(portfolio_variance) * np.sqrt(252)

    return annualized_volatility


def diversification_ratio(weights: pd.Series, covariance: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Calculate the diversification ratio of a portfolio.

    Parameters:
    -----------
    weights : pd.Series
        A Series representing the weights of the assets in the portfolio.
    covariance : np.ndarray or pd.DataFrame
        Either a NumPy array or a Pandas DataFrame representing the covariance matrix
        of the asset returns.

    Returns:
    --------
    np.ndarray
        An array representing the diversification ratio for each asset in the portfolio.

    Notes:
    ------
    The diversification ratio measures the contribution of each asset's volatility
    to the total portfolio volatility relative to its risk contribution.
    """
    p_volatility = portfolio_volatility(weights, covariance)
    asset_contributions = np.sqrt(np.diag(covariance.values)) * weights.values
    diversification_ratio = asset_contributions / p_volatility

    return diversification_ratio



def compute_maximum_drawdown(weights: pd.DataFrame,
                             prices: pd.DataFrame) -> pd.Series:
    """
    Compute the daily drawdown series and the maximum drawdown (MDD) for a portfolio.

    Parameters:
    ----------
    weights : pd.DataFrame
        A DataFrame where each column represents the weights of assets in the portfolio over time.
    prices : pd.DataFrame
        A DataFrame where each column represents the prices of the corresponding assets over time.

    Returns:
    -------
    pd.Series
        A Series containing the daily drawdown values for the portfolio.

    Notes:
    -----
    The function calculates the portfolio value by multiplying the weights with the prices and summing across assets.
    It then computes the drawdown as the percentage drop from the maximum value observed up to that point.
    """
    # Calculate the portfolio value
    portfolio_value = (weights * prices).sum(axis=1)

    # Calculate the running maximum
    running_max = portfolio_value.cummax()

    # Calculate the daily drawdown
    daily_drawdown = portfolio_value / running_max - 1.0

    # Return the daily drawdown series
    return daily_drawdown


import pandas as pd
import numpy as np


def portfolio_turnover(weights_today: pd.Series, weights_yesterday: pd.Series) -> float:
    """
    Calculate the daily portfolio turnover.

    Parameters:
    weights_today (pd.Series): Weights of assets today (time t).
    weights_yesterday (pd.Series): Weights of assets yesterday (time t-1).

    Returns:
    float: Portfolio turnover for the day.
    """
    # Calculate the absolute difference in portfolio weights
    turnover = np.abs(weights_today - weights_yesterday).sum() / 2
    return turnover


# Example data
weights_t = pd.Series({'Asset1': 0.3, 'Asset2': 0.5, 'Asset3': 0.2})  # Weights at time t
weights_t_minus_1 = pd.Series({'Asset1': 0.25, 'Asset2': 0.55, 'Asset3': 0.2})  # Weights at time t-1

# Calculate daily turnover
daily_turnover = portfolio_turnover(weights_t, weights_t_minus_1)
print(f'Daily Turnover: {daily_turnover:.4f}')
