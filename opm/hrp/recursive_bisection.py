
import numpy as np
import pandas as pd

def compute_cluster_variance(covariance: np.ndarray,
                                cluster_items: list)->float:
    """ Compute the variance of a cluster """
    # slice the covariance matrix
    covariance_slice = covariance.iloc[cluster_items, cluster_items]
    # calculate the inverse-variance portfolio
    ivp = 1./np.diag(covariance_slice)
    ivp/=ivp.sum()
    w_ = ivp.reshape(-1,1)
    cluster_variance = np.dot(np.dot(w_.T, covariance_slice), w_)[0,0]
    return cluster_variance


def compute_recursive_bisection_weights(covariance: np.ndarray,
                                        sort_ix: list)-> pd.Series:
    """ Compute the HRP allocation recursively
    """
    # intialize weights of 1
    weights = pd.Series(1, index=sort_ix)
    # intialize all items in one cluster
    cluster_items = [sort_ix]
    while len(cluster_items) > 0:
        # bisection
        """
        Example of bisection: 
        [[3, 6, 0, 9, 2, 4, 13, 5, 12, 8, 10, 7, 1, 11]]
        [[3, 6, 0, 9, 2, 4, 13], [5, 12, 8, 10, 7, 1, 11]]
        [[3, 6, 0], [9, 2, 4, 13], [5, 12, 8], [10, 7, 1, 11]]
        [[3], [6, 0], [9, 2], [4, 13], [5], [12, 8], [10, 7], [1, 11]]
        [[6], [0], [9], [2], [4], [13], [12], [8], [10], [7], [1], [11]]
        """
        cluster_items = [i[int(j):int(k)] for i in cluster_items for j, k in
                   ((0, len(i) / 2), (len(i) / 2, len(i))) if len(i) > 1]
        for i in range(0, len(cluster_items), 2):
            # cluster 1
            first_cluster_items = cluster_items[i]
            # cluster 2
            second_cluster_items = cluster_items[i + 1]
            # compute the variance of the two clusters
            first_cluster_variance = compute_cluster_variance(covariance, first_cluster_items)
            second_cluster_variance = compute_cluster_variance(covariance, second_cluster_items)
            # compute scaling factor
            alpha = 1 - first_cluster_variance / (first_cluster_variance + second_cluster_variance)
            # update the weights
            weights[first_cluster_items] *= alpha
            weights[second_cluster_items] *= 1 - alpha
    return weights

