# Optimal Portfolio Modeling

Author: Oualid Missaoui

## Scope

This repository provides implementations and demonstrations of various portfolio optimization algorithms:

- **Mean-Variance Optimization (MVO)**:
  - Includes implementations for minimum variance and maximum Sharpe ratio portfolios.
  - Utilizes the Critical Line Algorithm with a refactored implementation based on Baily and Lopez.

- **Hierarchical Risk Parity (HRP)**:
  - Implements Marcos Lopez de Prado's HRP algorithm, which uses hierarchical clustering to construct a tree of assets.
  - Computes risk parity weights for each cluster and subsequently for each asset.

- **Nested Clustered Optimization (NCO)**:
  - Based on De Prado's work, this algorithm optimizes portfolios using nested clustering approaches for enhanced diversification.

## Structure

The repository is organized as follows:

- **docs**: 
  - Contains notebooks explaining the underlying methodologies of each algorithm.
  - Contains bibliographic references.

- **opm**:
  - **cluster**: Utilities and algorithms for clustering assets.
  - **rp**: Implementation of the risk parity algorithm.
  - **hrp**: Implementation of the HRP algorithm.
  - **mvo**: Mean-variance optimization algorithms.
  - **nco**: Nested clustered optimization implementations.
  - **numerical_algorithms**: Core numerical algorithms used in portfolio optimization.

- **notebooks**: 
  - Example notebooks demonstrating the usage and results of the implemented algorithms.

## Bibliography

- Baily, R. E., & Lopez, J. A. (2005). A faster algorithm for portfolio optimization. *Management Science*, 51(11), 1676-1686.
- de Prado, M. L. (2016). Optimal portfolios from ordering information. *Journal of Portfolio Management*, 42(2), 65-80.
- de Prado, M. L. (2021). Advances in Financial Machine Learning. John Wiley & Sons.
