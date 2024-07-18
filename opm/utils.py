
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

