from typing import Union
import numpy as np
import pandas as pd
def compute_uniform_weights(variances: Union[np.ndarray, pd.Series],
                            security_names: list[str]) -> pd.Series:
    """
    Compute uniform weights for a list of securities.

    Parameters:
    -----------
    variances : Union[np.ndarray, pd.Series]
        Array or Series of variances (or any list of values) to determine the number of securities.
    security_names : List[str]
        List of security names to use as index for the resulting Series.

    Returns:
    --------
    pd.Series
        Series of uniform weights with security names as index.
    """
    if isinstance(variances, np.ndarray):
        variances = pd.Series(variances, index=security_names)
    elif isinstance(variances, pd.Series):
        variances = variances.reindex(security_names, fill_value=0.0)
    else:
        raise TypeError("Unsupported type for 'variances'. Expected np.ndarray or pd.Series.")

    weights = pd.Series(1 / len(variances), index=security_names, name="uniform")
    return weights

