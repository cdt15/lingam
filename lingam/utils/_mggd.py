import numpy as np
from scipy.stats import gamma
from scipy.special import digamma
from scipy.optimize import root_scalar


class MGGD:
    """Multivariate Generalized Gaussian Distribution (MGGD)"""

    def __init__(self, mean, cov, beta=1, tolerance=1e-6):
        """Initializes the MGGD with given parameters.

        Parameters
        ----------
        mean : array-like, shape (n_features,)
            Mean vector of the distribution.
        cov : array-like, shape (n_features, n_features)
            Covariance matrix of the distribution.
        beta : float, optional (default=1)
            Shape parameter of the distribution.
        tolerance : float, optional (default=1e-6)
            Tolerance for convergence in parameter estimation.
        """
        self.mean = np.array(mean)
        self.cov = np.array(cov)
        self.beta = beta
        self.tolerance = tolerance

        n_features = self.mean.shape[0]

        if self.cov.shape[0] != n_features or self.cov.shape[1] != n_features:
            raise ValueError("cov must be a square matrix matching the length of mean.")
        if not np.allclose(self.cov, self.cov.T):
            raise ValueError("cov must be a symmetric matrix.")
        if np.any(
            np.linalg.eigvals(self.cov)
            < self.tolerance * np.max(np.abs(np.linalg.eigvals(self.cov)))
        ):
            raise ValueError("cov must be a symmetric positive definite matrix.")
        if self.beta <= 0:
            raise ValueError("beta must be a positive value.")

    def rvs(self, size):
        """Generates random samples from the MGGD.

        Parameters
        ----------
        size : int
            Number of samples to generate.

        Returns
        -------
        samples : array-like, shape (size, n_features)
            Generated samples.
        """
        p = self.mean.shape[0]
        eigvals, eigvecs = np.linalg.eigh(self.cov)
        A = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        r = gamma.rvs(a=p / (2 * self.beta), scale=2, size=size) ** (
            1 / (2 * self.beta)
        )
        U = np.random.multivariate_normal(np.zeros(p), np.eye(p), size)
        U_norm = np.linalg.norm(U, axis=1, keepdims=True)
        U = U / U_norm

        samples = (r[:, np.newaxis] * (U @ A.T)) + self.mean
        return samples


class MGGDEstimator:
    """Estimator for Multivariate Generalized Gaussian Distribution (MGGD) parameters"""

    def __init__(self):
        pass

    def fit(self, X, eps=1e-6):
        """Estimates the parameters of the MGGD from data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        eps : float, optional (default=1e-6)
            Small value to avoid division by zero.

        Returns
        -------
        mggd : MGGD
            Estimated MGGD object.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n, p = X.shape
        mean = np.mean(X, axis=0)
        X_centered = X - mean

        cov = np.eye(p)
        beta = 0.1
        beta_1 = np.inf

        k = 0
        while np.abs(beta - beta_1) > eps and k < 1000:
            k += 1
            cov_k = np.zeros((p, p))
            inv_cov = np.dot(X_centered, np.linalg.inv(cov))
            u = np.sum(inv_cov * X_centered, axis=1)

            for i in range(n):
                cov_k += u[i] ** (beta - 1) * np.outer(X_centered[i], X_centered[i])

            cov = cov_k / n
            cov = (p / np.sum(np.diag(cov))) * cov

            beta_1 = beta
            beta = self._estimate_beta(u, beta, p, eps)

        m = (beta / (p * n) * np.sum(u**beta)) ** (1 / beta)
        cov *= m

        return MGGD(mean=mean, cov=cov, beta=beta, tolerance=eps)

    def _estimate_beta(self, u, beta, p, eps):
        """Estimates the shape parameter beta using root finding."""
        N = len(u)

        def equation(z):
            term1 = p * N / (2 * np.sum(u**z)) * np.sum(np.log(u + eps) * u**z)
            term2 = p * N / (2 * z) * (np.log(2) + digamma(p / (2 * z)))
            term3 = N
            term4 = p * N / (2 * z) * np.log(z / (p * N) * np.sum(u**z + eps))
            return term1 - term2 - term3 - term4

        bracket = [eps, 2 * np.ceil(beta)]
        f_a, f_b = equation(bracket[0]), equation(bracket[1])

        while f_a * f_b > 0:
            if f_a > 0:
                bracket[0] /= 2
            else:
                bracket[1] *= 2
            f_a, f_b = equation(bracket[0]), equation(bracket[1])

        result = root_scalar(equation, bracket=bracket, method="brentq")
        return result.root
