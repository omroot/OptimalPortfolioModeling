from typing import Union, Optional, List, Dict, Any, Tuple
from multiprocessing import Pool
from joblib import Parallel, delayed
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


class OptimalClustering:
    """"A class for optimal clustering that automatically determines the number of clusters.
    The base clustering method is k-means.
    The optimal number of clusters is determined based on the silhouette method [1] or the gap-statistics [2].

    References:
    ----------
    [1] Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis.
    [2] Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the number of clusters in a data set via the gap statistic.
    [3] Marcos Lopez de Prado, Machine learning for Asset Managers, Cambridge elements , 2020.

    """
    def __init__(
            self,
            is_input_correlation: bool = True,
            max_number_clusters: Optional[int] = None,
            number_initializations: int = 10,
            n_jobs: int = -1,  # Number of parallel jobs, -1 means using all available cores
            method: str = 'silhouette',  # Method to determine optimal clusters ('silhouette' or 'gap_statistic')
            use_robust_quality: bool = False,  # Use robust quality metric
            number_references: int = 20,  # Number of reference datasets for gap statistic
            **kwargs: Any
    ):
        """Constructs all the neessary attributes for the OptimalClustering object.

        Args:
            is_input_correlation (bool): A flag to indicate if the input data is a correlation matrix.
            max_number_clusters (int, optional): The maximum number of clusters.
            number_initializations (int): The number of initializations.
            n_jobs (int): Number of parallel jobs for fitting.
            method (str): Method to determine the optimal number of clusters ('silhouette' or 'gap_statistic').
            use_robust_quality (bool): Use robust quality metric.
            number_references (int): Number of reference datasets for gap statistic.
        """
        self.is_input_correlation = is_input_correlation
        self.max_number_clusters = max_number_clusters
        self.number_initializations = number_initializations
        self.method = method
        self.number_references = number_references
        self.use_robust_quality = use_robust_quality
        self.columns: Optional[List[str]] = None
        self.quality: Optional[float] = None
        self.silhouette: Optional[np.ndarray] = None
        self.reordered_X: Optional[pd.DataFrame] = None
        self.gaps: Optional[np.ndarray] = None
        self.n_jobs = n_jobs
        self.kwargs = kwargs

    def _fit_wrapper(self, k: int, X: pd.DataFrame) -> Tuple[
        int, np.ndarray, np.ndarray, float, int, np.ndarray, float]:
        km = KMeans(n_clusters=k, n_init=self.number_initializations, **self.kwargs)
        km.fit(X)
        silhouette = silhouette_samples(X, km.labels_)
        if not self.use_robust_quality:
            quality = silhouette.mean() / silhouette.std()
        else:
            quality = pd.Series(silhouette).median() / median_abs_deviation(silhouette)

        return k, km.cluster_centers_, km.labels_, km.inertia_, km.n_iter_, silhouette, quality

    def _calculate_gap_statistic(self, X: pd.DataFrame) -> Tuple[int, np.ndarray, np.ndarray, float, int]:
        def compute_inertia_and_labels(k: int, data: np.ndarray) -> Tuple[float, np.ndarray]:
            km = KMeans(n_clusters=k, n_init=self.number_initializations, **self.kwargs)
            km.fit(data)
            return km.inertia_, km.labels_

        gaps = np.zeros((self.max_number_clusters,))
        reference_dispersions_std = np.zeros((self.max_number_clusters,))

        for k in range(1, self.max_number_clusters + 1):
            reference_dispersions = np.zeros(self.number_references)
            for i in range(self.number_references):
                reference_dataset = np.random.random_sample(size=X.shape)
                reference_dispersion, _ = compute_inertia_and_labels(k, reference_dataset)
                reference_dispersions[i] = reference_dispersion

            original_dispersion, _ = compute_inertia_and_labels(k, X)
            gaps[k-1] = np.mean(np.log(reference_dispersions)) - np.log(original_dispersion)
            reference_dispersions_std[k-1] = np.sqrt(1+1/self.number_references) * np.std(np.log(reference_dispersions))
        self.gaps=gaps
        try:
            optimal_number_of_clusters =  (
                np.where(
                    gaps[:-1] >= gaps[1:] - reference_dispersions_std[1:]
                )[0][0] + 1
            )
        except:
            optimal_number_of_clusters = 1
        optimal_gap = gaps[optimal_number_of_clusters - 1]
        optimal_inertia, optimal_labels = compute_inertia_and_labels(optimal_number_of_clusters, X)
        return optimal_number_of_clusters, optimal_labels, optimal_inertia,  optimal_gap

    def fit(self, X: pd.DataFrame,
            y: Optional[Union[np.ndarray, pd.Series]] = None,
            sample_weight: Optional[np.ndarray] = None) -> None:
        """Fit the clustering model to the data.

        Args:
            X (pd.DataFrame): Input data for clustering.
            y (Optional[Union[np.ndarray, pd.Series]]): Target labels.
            sample_weight (Optional[np.ndarray]): Sample weights.
        """
        self.columns = X.columns

        if self.is_input_correlation:
            X = ((1 - X.fillna(0)) / 2) ** 0.5

        if self.max_number_clusters is None:
            self.max_number_clusters = X.shape[1] - 1

        if self.method == 'gap_statistic':
            print(f'Clustering using Gap statistic')
            self.n_clusters, self.labels_, self.inertia_, self.quality = self._calculate_gap_statistic(X)
        else:
            print(f'Clustering using the Silhouette method')
            results = []
            # combinations = product(range(self.number_initializations), range(2, self.max_number_clusters + 1))
            # results = Parallel(n_jobs=self.n_jobs)(
            #     delayed(self._fit_wrapper)(k, X) for init, k in combinations
            # )
            for init in range(self.number_initializations):
                results.extend(
                    Parallel(n_jobs=self.n_jobs)(
                        delayed(self._fit_wrapper)(k, X) for k in range(2, self.max_number_clusters + 1)
                    )
                )
            # with Pool(self.n_jobs) as pool:
            #     for init in range(self.number_initializations):
            #         results.extend(
            #             pool.starmap(self._fit_wrapper, [(k, X) for k in range(2, self.max_number_clusters + 1)]))

            optimal_cluster_centers_ = None
            optimal_labels_ = None
            optimal_inertia_ = None
            optimal_n_iter_ = None
            optimal_silhouette = None
            optimal_quality = -1

            for k, cluster_centers_, labels_, inertia_, n_iter_, silhouette, quality in results:
                if optimal_silhouette is None or quality > optimal_quality:
                    optimal_silhouette = silhouette
                    optimal_cluster_centers_ = cluster_centers_
                    optimal_labels_ = labels_.copy()
                    optimal_inertia_ = inertia_
                    optimal_n_iter_ = n_iter_
                    optimal_quality = quality

            self.cluster_centers_ = optimal_cluster_centers_
            self.labels_ = optimal_labels_
            self.inertia_ = optimal_inertia_
            self.n_iter_ = optimal_n_iter_
            self.quality = optimal_quality
            self.silhouette = optimal_silhouette
            self.n_clusters = len(np.unique(self.labels_))

        new_indices = np.argsort(self.labels_)
        reordered_X = X.iloc[new_indices]  # reorder rows
        reordered_X = reordered_X.iloc[:, new_indices]  # reorder columns

        self.reordered_X = reordered_X

    @property
    def clusters(self) -> Dict[int, List[str]]:
        """Get clusters with their members.

        Returns:
            Dict[int, List[str]]: Cluster number as key and list of cluster members as value.
        """
        return {i: self.columns[np.where(self.labels_ == i)[0]].tolist()
                for i in np.unique(self.labels_)}
