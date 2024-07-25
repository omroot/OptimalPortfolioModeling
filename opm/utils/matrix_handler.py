import numpy as np
import pandas as pd


class CovarianceMatrixHandler:
    def __init__(self, cov_matrix=None, df_variances=None, df_covariances=None):
        if cov_matrix is not None:
            self.cov_matrix = cov_matrix
            self.df_variances, self.df_covariances = self._matrix_to_dataframes(cov_matrix)
        elif df_variances is not None and df_covariances is not None:
            self.df_variances = df_variances
            self.df_covariances = df_covariances
            self.cov_matrix = self._dataframes_to_matrix(df_variances, df_covariances)
        else:
            raise ValueError("Either cov_matrix or df_variances and df_covariances must be provided")

    def _matrix_to_dataframes(self, cov_matrix):
        # Extract variances (diagonal elements)
        variances = np.diag(cov_matrix)
        df_variances = pd.DataFrame({
            'item': range(len(variances)),
            'variance': variances
        })

        # Extract covariances (off-diagonal elements)
        cov_data = []
        for i in range(cov_matrix.shape[0]):
            for j in range(i + 1, cov_matrix.shape[1]):
                cov_data.append([i, j, cov_matrix[i, j]])

        df_covariances = pd.DataFrame(cov_data, columns=['item1', 'item2', 'covariance'])

        return df_variances, df_covariances

    def _dataframes_to_matrix(self, df_variances, df_covariances):
        num_items = df_variances.shape[0]
        cov_matrix = np.zeros((num_items, num_items))

        # Fill the diagonal with variances
        for index, row in df_variances.iterrows():
            cov_matrix[row['item'], row['item']] = row['variance']

        # Fill the off-diagonal elements with covariances
        for index, row in df_covariances.iterrows():
            i, j, cov = row['item1'], row['item2'], row['covariance']
            cov_matrix[i, j] = cov
            cov_matrix[j, i] = cov  # Covariance matrix is symmetric

        return cov_matrix

    def get_cov_matrix(self):
        return self.cov_matrix

    def get_variances_df(self):
        return self.df_variances

    def get_covariances_df(self):
        return self.df_covariances

    def set_cov_matrix(self, cov_matrix):
        self.cov_matrix = cov_matrix
        self.df_variances, self.df_covariances = self._matrix_to_dataframes(cov_matrix)

    def set_dataframes(self, df_variances, df_covariances):
        self.df_variances = df_variances
        self.df_covariances = df_covariances
        self.cov_matrix = self._dataframes_to_matrix(df_variances, df_covariances)


# Example usage
cov_matrix = np.array([
    [1.0, 0.8, 0.3],
    [0.8, 1.0, 0.5],
    [0.3, 0.5, 1.0]
])

handler = CovarianceMatrixHandler(cov_matrix=cov_matrix)
print("Covariance Matrix:")
print(handler.get_cov_matrix())
print("\nVariances DataFrame:")
print(handler.get_variances_df())
print("\nCovariances DataFrame:")
print(handler.get_covariances_df())

# Reconstruct from DataFrames
df_variances = handler.get_variances_df()
df_covariances = handler.get_covariances_df()
handler_reconstructed = CovarianceMatrixHandler(df_variances=df_variances, df_covariances=df_covariances)
print("\nReconstructed Covariance Matrix:")
print(handler_reconstructed.get_cov_matrix())
