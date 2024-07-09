from typing import Union
import numpy as np
import pandas as pd

def compute_risk_parity_weights(covariances: Union[np.ndarray, pd.DataFrame],
                       security_names: list) -> pd.Series:
    """
    Compute Risk Parity (RP) weights for a given covariance matrix.

    Args:
    - covariances (Union[np.ndarray, pd.DataFrame]): Covariance matrix of the securities.
      If pd.DataFrame, it's assumed to have security names as both index and columns.
    - security_names (list): List of security names, used as index for the output Series.

    Returns:
    - pd.Series: Series of Risk Parity (RP) weights, indexed by security names.
    """
    if isinstance(covariances, pd.DataFrame):
        covariances = covariances.values

    weights = 1 / np.diag(covariances)
    normalized_weights = weights / np.sum(weights)
    return pd.Series(normalized_weights, index=security_names, name="risk_parity")
