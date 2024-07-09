
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
from typing import Union


def covariance2correlation(covariance_matrix: np.ndarray)-> np.ndarray:
    """Converts a covariance matrix into a correlation matrix
    Args:
        covariance_matrix: input covariance matrix
    Returns:
        correlation_matrix: output correlation matrix
    """
    # Derive the correlation matrix from a covariance matrix
    std=np.sqrt(np.diag(covariance_matrix))
    correlation_matrix=covariance_matrix/np.outer(std,std)
    correlation_matrix[correlation_matrix<-1] = -1 # numerical error
    correlation_matrix[correlation_matrix>1]  = 1 # numerical error
    return correlation_matrix


def correlation_to_covariance(correlation_matrix: np.ndarray,
                              standard_deviations: np.ndarray) -> np.ndarray:
    """
    Convert a correlation matrix to a covariance matrix.

    Parameters:
    ----------
    correlation_matrix : np.ndarray
        A square matrix representing the correlation coefficients between variables.
    standard_deviations : np.ndarray
        A 1D array of standard deviations for each variable.

    Returns:
    -------
    np.ndarray
        The covariance matrix.

    Notes:
    -----
    The covariance matrix is calculated by multiplying the correlation matrix
    element-wise with the outer product of the standard deviations.
    """
    covariance_matrix = correlation_matrix * np.outer(standard_deviations, standard_deviations)
    return covariance_matrix


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

