
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from opm.numerical_algorithms import golden_section
class CriticalLineAlgorithm:
    def __init__(self, mean: np.ndarray,
                 covariance: np.ndarray,
                 lower_bounds: np.ndarray,
                 upper_bounds: np.ndarray):
        """
        Initialize the Critical Line Algorithm.

        :param mean: Mean returns for each asset.
        :param covariance: Covariance matrix of asset returns.
        :param lower_bounds: Lower bounds for asset weights.
        :param upper_bounds: Upper bounds for asset weights.
        """
        self.mean = mean
        self.covariance = covariance
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.weights: List[np.ndarray] = []  # Solutions
        self.lambdas: List[Optional[float]] = []  # Lambdas
        self.gammas: List[Optional[float]] = []  # Gammas
        self.free_weights: List[List[int]] = []  # Free weights

    def solve(self) -> None:
        """
        Compute the turning points, free sets, and weights.
        """
        free, weights = self._initialize_algorithm()
        self.weights.append(np.copy(weights))  # Store solution
        self.lambdas.append(None)
        self.gammas.append(None)
        self.free_weights.append(free[:])

        while True:
            lambda_in, index_in, bound_in = None, None, None
            if len(free) > 1:
                covar_f, covar_fb, mean_f, weights_b = self._get_matrices(free)
                covar_f_inv = np.linalg.inv(covar_f)
                for idx, i in enumerate(free):
                    lambda_val, bound = self._compute_lambda(covar_f_inv, covar_fb, mean_f, weights_b, idx,
                                                             [self.lower_bounds[i], self.upper_bounds[i]])
                    if lambda_in is None or lambda_val > lambda_in:
                        lambda_in, index_in, bound_in = lambda_val, i, bound

            lambda_out, index_out = None, None
            if len(free) < self.mean.shape[0]:
                bounded = self._get_bounded(free)
                for i in bounded:
                    covar_f, covar_fb, mean_f, weights_b = self._get_matrices(free + [i])
                    covar_f_inv = np.linalg.inv(covar_f)
                    lambda_val, bound = self._compute_lambda(covar_f_inv,
                                                             covar_fb,
                                                             mean_f,
                                                             weights_b,
                                                             mean_f.shape[0] - 1,
                                                             self.weights[-1][i]
                                                             )
                    if (self.lambdas[-1] is None or lambda_val < self.lambdas[-1]) and (
                            lambda_out is None or lambda_val > lambda_out):
                        lambda_out, index_out = lambda_val, i

            if (lambda_in is None or lambda_in < 0) and (lambda_out is None or lambda_out < 0):
                break

            if lambda_in is not None and (lambda_out is None or lambda_in > lambda_out):
                self.lambdas.append(lambda_in)
                free.remove(index_in)
                weights[index_in] = bound_in  # Set value at the correct boundary
            else:
                self.lambdas.append(lambda_out)
                free.append(index_out)

            covar_f, covar_fb, mean_f, weights_b = self._get_matrices(free)
            covar_f_inv = np.linalg.inv(covar_f)
            weights_f, gamma = self._compute_weights(covar_f_inv, covar_fb, mean_f, weights_b)
            for i in range(len(free)):
                weights[free[i]] = weights_f[i]

            self.weights.append(np.copy(weights))  # Store solution
            self.gammas.append(gamma)
            self.free_weights.append(free[:])

            if len(free) == self.mean.shape[0]:
                weights_f, gamma = self._compute_weights(covar_f_inv, covar_fb, np.zeros(mean_f.shape), weights_b)
                for i in range(len(free)):
                    weights[free[i]] = weights_f[i]
                self.weights.append(np.copy(weights))  # Store solution
                self.gammas.append(gamma)
                self.free_weights.append(free[:])

    def _initialize_algorithm(self) -> Tuple[List[int], np.ndarray]:
        """
        Initialize the algorithm.

        :return: A tuple containing the first free weight and the initial weights vector.
        """
        # Form Structured array
        a = np.zeros((self.mean.shape[0]), dtype=[('id', int), ('mu', float)])
        b = [self.mean[i][0] for i in range(self.mean.shape[0])]
        a[:] = list(zip(range(self.mean.shape[0]), b))
        # sort structured array
        b = np.sort(a, order='mu')
        # first free weight
        i, weights = b.shape[0], np.copy(self.lower_bounds)
        while np.sum(weights) < 1:
            i -= 1
            weights[b[i][0]] = self.upper_bounds[b[i][0]]
        weights[b[i][0]] += 1 - np.sum(weights)
        return [b[i][0]], weights

    def _compute_boundary(self,
                          condition: float,
                          bounds: List[float]) -> float:
        """
        Compute the boundary value based on the condition.

        :param condition: Condition to determine the boundary.
        :param bounds: List of bounds to choose from.
        :return: The boundary value.
        """
        if condition > 0:
            return bounds[1]
        if condition < 0:
            return bounds[0]
        return bounds[0]

    def _compute_weights(self,
                         covar_f_inv: np.ndarray,
                         covar_fb: np.ndarray,
                         mean_f: np.ndarray,
                         weights_b: Optional[np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Compute the weights and gamma.

        :param covar_f_inv: Inverse of the covariance matrix of free assets.
        :param covar_fb: Covariance matrix of free and bounded assets.
        :param mean_f: Mean returns of free assets.
        :param weights_b: Weights of bounded assets.
        :return: A tuple containing the weights and gamma.
        """
        ones_f = np.ones(mean_f.shape)
        g1 = np.dot(np.dot(ones_f.T, covar_f_inv), mean_f)
        g2 = np.dot(np.dot(ones_f.T, covar_f_inv), ones_f)
        if weights_b is None:
            gamma = float(-self.lambdas[-1] * g1 / g2 + 1 / g2)
            weights_adjustment = 0
        else:
            ones_b = np.ones(weights_b.shape)
            g3 = np.dot(ones_b.T, weights_b)
            g4 = np.dot(covar_f_inv, covar_fb)
            weights_adjustment = np.dot(g4, weights_b)
            g4 = np.dot(ones_f.T, weights_adjustment)
            gamma = float(-self.lambdas[-1] * g1 / g2 + (1 - g3 + g4) / g2)
        weights_final = np.dot(covar_f_inv, ones_f)
        weights_mean = np.dot(covar_f_inv, mean_f)
        return -weights_adjustment + gamma * weights_final + self.lambdas[-1] * weights_mean, gamma

    def _compute_lambda(self,
                        covar_f_inv: np.ndarray,
                        covar_fb: np.ndarray,
                        mean_f: np.ndarray,
                        weights_b: Optional[np.ndarray],
                        index: int,
                        bounds: Any) -> Tuple[Optional[float], Any]:
        """
        Compute the lambda value.

        :param covar_f_inv: Inverse of the covariance matrix of free assets.
        :param covar_fb: Covariance matrix of free and bounded assets.
        :param mean_f: Mean returns of free assets.
        :param weights_b: Weights of bounded assets.
        :param index: Index of the asset.
        :param bounds: Bounds for the asset weights.
        :return: A tuple containing the lambda value and the boundary.
        """
        ones_f = np.ones(mean_f.shape)
        c1 = np.dot(np.dot(ones_f.T, covar_f_inv), ones_f)
        c2 = np.dot(covar_f_inv, mean_f)
        c3 = np.dot(np.dot(ones_f.T, covar_f_inv), mean_f)
        c4 = np.dot(covar_f_inv, ones_f)
        condition = -c1 * c2[index] + c3 * c4[index]
        if condition == 0:
            return None, None
        if isinstance(bounds, list):
            bounds = self._compute_boundary(condition, bounds)
        if weights_b is None:
            return float((c4[index] - c1 * bounds) / condition), bounds
        else:
            ones_b = np.ones(weights_b.shape)
            l1 = np.dot(ones_b.T, weights_b)
            l2 = np.dot(covar_f_inv, covar_fb)
            l3 = np.dot(l2, weights_b)
            l2 = np.dot(ones_f.T, l3)
            return float(((1 - l1 + l2) * c4[index] - c1 * (bounds + l3[index])) / condition), bounds

    def _get_matrices(self, free: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Get the covariance matrices and mean returns for free and bounded assets.

        :param free: List of indices for free assets.
        :return: A tuple containing covariance matrices and mean returns for free and bounded assets.
        """
        n = self.mean.shape[0]
        bound = self._get_bounded(free)
        covar_f = np.zeros((len(free), len(free)))
        covar_fb = np.zeros((len(free), len(bound)))
        mean_f = np.zeros((len(free), 1))
        weights_b = np.zeros((len(bound), 1))
        for i in range(n):
            if i in free:
                j = free.index(i)
                mean_f[j, 0] = self.mean[i, 0]
                for k in range(n):
                    if k in free:
                        l = free.index(k)
                        covar_f[j, l] = self.covariance[i, k]
                    else:
                        l = bound.index(k)
                        covar_fb[j, l] = self.covariance[i, k]
            else:
                j = bound.index(i)
                weights_b[j, 0] = self.weights[-1][i]
        return covar_f, covar_fb, mean_f, weights_b

    def _get_bounded(self, free: List[int]) -> List[int]:
        """
        Get the indices of bounded assets.

        :param free: List of indices for free assets.
        :return: List of indices for bounded assets.
        """
        return [i for i in range(self.mean.shape[0]) if i not in free]

    def extract_portfolios(self) -> Tuple[List[Optional[float]], List[np.ndarray]]:
        """
        Extract weights and gammas for the portfolios along the efficient frontier.

        :return: A tuple containing the list of gammas and the list of weights.
        """
        return self.gammas, self.weights

    def extract_efficient_frontier(self, points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the efficient frontier as points of expected returns and risks.

        :param points: Number of points on the efficient frontier.
        :return: A tuple containing the array of risks and expected returns.
        """
        # Get the efficient frontier
        mu, sigma, weights = [], [], []
        num_points = int(points / len(self.weights))
        b = range(len(self.weights) - 1)

        for i in b:
            w0, w1 = self.weights[i], self.weights[i + 1]
            if i == b[-1]:
                a = np.linspace(0, 1, int(points / len(self.weights)))  # Cast points to int
            else:
                a = np.linspace(0, 1, num_points)[:-1]  # Ensure points / len(self.w) is cast to int

            for j in a:
                w = w1 * j + (1 - j) * w0
                # Compute mu, sigma, weights based on w and append to respective lists
                weights.append(np.copy(w))
                mu.append(np.dot(w.T,self.mean)[0,0])
                sigma.append(np.dot(np.dot(w.T,self.covariance),w)[0,0]**.5)

        return mu, sigma, weights
    def get_minimum_variance_portfolio(self):
        # Get the minimum variance solution
        var=[]
        for w in self.weights:
            a=np.dot(np.dot(w.T,self.covariance),w)
            var.append(a)
        return min(var)**.5,self.weights[var.index(min(var))]





    def get_sharpe_of_convex_combination(self,
                                         alpha: float,
                                         first_portfolio_weights: np.ndarray,
                                         second_portfolio_weights: np.ndarray) -> float:
        """
        Evaluate Sharpe Ratio of the portfolio within the convex combination of two weight vectors.

        Args:
        - alpha (float): Convex combination parameter (0 <= alpha <= 1).
        - first_portfolio_weights (np.ndarray): First weight vector.
        - second_portfolio_weights (np.ndarray): Second weight vector.

        Returns:
        - float: Sharpe Ratio of the portfolio.
        """
        assert 0 <= alpha <= 1, "Parameter 'alpha' must be within the range [0, 1]."
        assert first_portfolio_weights.shape == second_portfolio_weights.shape, "Weight vectors must have the same shape."
        assert first_portfolio_weights.shape[0] == self.mean.shape[
            0], "Weight vector size must match the mean vector size."
        assert first_portfolio_weights.shape[0] == self.covariance.shape[
            0], "Weight vector size must match the covariance matrix size."

        combined_weights = alpha * first_portfolio_weights + (
                    1 - alpha) * second_portfolio_weights  # Convex combination of weights
        expected_return = np.dot(combined_weights.T, self.mean)[0, 0]  # Expected portfolio return
        portfolio_std_dev = np.dot(np.dot(combined_weights.T, self.covariance), combined_weights)[
                                0, 0] ** 0.5  # Portfolio standard deviation
        sharpe_ratio = expected_return / portfolio_std_dev  # Sharpe Ratio calculation
        return sharpe_ratio

    def get_maximum_sharpe_portfolio(self):
        # Get the max Sharpe ratio portfolio
        #  Compute the local max SR portfolio between any two neighbor turning points
        w_sr,sr=[],[]
        for i in range(len(self.weights)-1):
            w0=np.copy(self.weights[i])
            w1=np.copy(self.weights[i+1])
            kargs={'minimum':False,
                   'args':(w0,w1)}
            a,b= golden_section(obj=self.get_sharpe_of_convex_combination,
                                a= 0,
                                b=1,
                                **kargs)


            w_sr.append(a*w0+(1-a)*w1)
            sr.append(b)
        return max(sr),w_sr[sr.index(max(sr))]

