from typing import Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

class HierarchicalClustering:
    def __init__(self, is_similarity_matrix: bool = True):
        """
        Initialize the HierarchicalClustering class.

        :param is_similarity_matrix: Indicates whether the input matrix is a similarity matrix (e.g., correlation matrix).
        """
        self.is_similarity_matrix = is_similarity_matrix
        self.labels = None
        self.pairwise_distance_matrix = None
        self.linkage_matrix = None
        self.dendrogram_data = None

    def build_distance_matrix(self, correlation_matrix: pd.DataFrame) -> None:
        """
        Build the distance matrix based on the correlation matrix.

        :param correlation_matrix: A DataFrame representing the correlation matrix.
        """
        self.pairwise_distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix))

    def compute_linkage(self, method='single') -> None:
        """
        Compute the linkage matrix using the specified method.

        :param method: Method for hierarchical clustering (default is 'single').
        """
        if self.pairwise_distance_matrix is None:
            raise ValueError("Distance matrix not computed. Call build_distance_matrix first.")
        self.linkage_matrix = linkage(self.pairwise_distance_matrix, method)
        self.dendrogram_data = pd.DataFrame(self.linkage_matrix)

    def fit(self, X: pd.DataFrame, method: str = 'single') -> None:
        """
        Fit the hierarchical clustering model by building the distance matrix and computing the linkage.

        :param X: Input DataFrame, either a correlation matrix or raw data.
        :param method: Method for hierarchical clustering (default is 'single').
        """
        self.labels = X.columns
        if self.is_similarity_matrix:
            self.build_distance_matrix(X)
        else:
            similarity_matrix = X.corr()
            self.build_distance_matrix(similarity_matrix)
        self.compute_linkage(method)

    def plot_dendrogram(self, figsize: Tuple[int, int] = (25, 10)) -> None:
        """
        Plot the dendrogram based on the linkage matrix.

        :param figsize: Size of the figure for the dendrogram plot (default is (25, 10)).
        """
        if self.linkage_matrix is None:
            raise ValueError("Linkage matrix not computed. Call fit first.")

        fig = plt.figure(figsize=figsize)
        self.dendrogram_data = dendrogram(self.dendrogram_data, labels=self.labels)
        plt.show()

# Example usage
if __name__ == "__main__":
    corr = pd.DataFrame(...)  # Your correlation matrix here
    etf_names = [...]  # Your ETF names here

    hc = HierarchicalClustering(is_similarity_matrix=True)
    hc.fit(corr, method='single')
    hc.plot_dendrogram()
