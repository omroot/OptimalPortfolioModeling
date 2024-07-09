
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from opm.cluster import OptimalClustering
from opm.mvo.maximum_sharpe import get_maximum_sharpe_portfolio_weights
from opm.mvo.minimum_variance import get_minimum_variance_portfolio_weights

from opm.utils import covariance2correlation



def nco(covariance: np.ndarray,
        mu: np.ndarray =None,
        max_number_clusters=None,
        n_jobs = 10):
    """ The nested clustering optimization """
    covariance =pd.DataFrame(covariance)
    if mu is not None:
        mu =pd.Series(mu[: ,0])
    correlation = covariance2correlation(covariance)

    optimal_clustering_model = OptimalClustering(n_jobs=n_jobs,
                                                 method = 'silhouette',
                                                 is_input_correlation=False,
                                                 max_number_clusters=max_number_clusters)
    optimal_clustering_model.fit(correlation)
    clusters = optimal_clustering_model.clusters


    print(f'Optimal number of clusters identified : {optimal_clustering_model.n_clusters}')
    # Initialize the intra-clusters weights matrix
    intracluster_weights =pd.DataFrame(0,
                                        index=covariance.index,
                                        columns=clusters.keys())

    for c, c_indicies in clusters.items():
        _cov_matrix = covariance.loc[c_indicies,c_indicies].values
        if mu is None:
            _pt_weights = get_minimum_variance_portfolio_weights(_cov_matrix)
        else:
            _mu = mu.loc[c_indicies].values.reshape(-1, 1)
            _pt_weights = get_maximum_sharpe_portfolio_weights(_cov_matrix, _mu)
        intracluster_weights.loc[c_indicies, c] = _pt_weights

    reduced_covariance_matrix = intracluster_weights.T.dot(np.dot(covariance, intracluster_weights))
    reduced_mu = (None if mu is None else intracluster_weights.T.dot(mu))

    if reduced_mu is None:
        intercluster_weights = pd.Series(get_minimum_variance_portfolio_weights(reduced_covariance_matrix).flatten(),
                                         index=reduced_covariance_matrix.index)
    else:
        intercluster_weights = pd.Series(
            get_maximum_sharpe_portfolio_weights(reduced_covariance_matrix,
                                                 reduced_mu).flatten(),
            index=reduced_covariance_matrix.index)
    nco_portfolio_weights = intracluster_weights.mul(intercluster_weights, axis=1).sum(axis=1).sort_index()
    return nco_portfolio_weights

