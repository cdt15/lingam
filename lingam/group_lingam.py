"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""

import itertools
import numbers

import numpy as np
from scipy.stats.distributions import chi2
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array, resample

from .utils import predict_adaptive_lasso, calculate_total_effect
from .hsic import hsic_test_gamma
from .bootstrap import BootstrapResult


class GroupLiNGAM:
    """Implementation of GroupLiNGAM Algorithm [1]_

    References
    ----------
    .. [1] Y. Kawahara, K. Bollen, S. Shimizu and T. Washio.
       GroupLiNGAM: Linear non-Gaussian acyclic models for sets of variables.
       Arxiv preprint arXiv:1006.5041, 2010.
    """
    def __init__(self, alpha=0.01):
        """Construct a GroupLiNGAM model.

        Parameters
        ----------
        alpha : float, optional (default=0.01)
            Alpha level for HSIC independence test.
        """
        # Check parameters
        if alpha < 0:
            raise ValueError("alpha must be >= 0.")

        self._alpha = alpha
        self._adjacency_matrix = None
        self._causal_order = []

    def fit(self, X):
        """Fit the model to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Check parameters
        X = check_array(X)
        n_features = X.shape[1]

        # Causal discovery
        V = list(np.arange(n_features))
        self._causal_order = self._group_search(V, X)

        return self._estimate_adjacency_matrix(X)

    def _compute_residuals(self, predictors, target):
        """Compute residuals"""
        lr = LinearRegression().fit(predictors, target)
        resid = target - lr.predict(predictors)
        return resid

    def _fisher_hsic_test(self, X, Y):
        """Conduct statistical test by HSIC and Fisher's method for multivariate X and Y."""
        fisher_stat = 0
        n_features_X = X.shape[1]
        n_features_Y = Y.shape[1]

        for i in range(n_features_X):
            for j in range(n_features_Y):
                _, hsic_p = hsic_test_gamma(X[:, [i]], Y[:, [j]])
                fisher_stat += np.inf if hsic_p == 0 else -2 * np.log(hsic_p)

        df = 2 * n_features_X * n_features_Y
        fisher_p = chi2.sf(fisher_stat, df=df)

        return fisher_stat, fisher_p

    def _group_search(self, U, X_U):
        """Recursively devide the given subset U into independent groups"""
        # Generate the subsets
        subsets = []
        if len(U) == 1:
            return [U]
        else:
            for num in range(1, len(U)):
                subsets.extend([list(comb) for comb in itertools.combinations(U, num)])

        hsic_p_list = []
        for S in subsets:
            # Compute the residuals
            U_S = np.setdiff1d(U, S)
            rS = self._compute_residuals(X_U[:, S], X_U[:, U_S])

            # Compute some independence
            _, hsic_p = self._fisher_hsic_test(X_U[:, S], rS)
            hsic_p_list.append(hsic_p)

        S_star_idx = np.argmax(hsic_p_list)
        S_star = subsets[S_star_idx]
        hsic_p_S_star = hsic_p_list[S_star_idx]

        K = []
        if hsic_p_S_star > self._alpha and len(U) > 1:
            # Compute the residuals
            U_S_star = np.setdiff1d(U, S_star)
            rS_star = self._compute_residuals(X_U[:, S_star], X_U[:, U_S_star])

            # Call the GroupSearch function recursively
            index_map = {val: i for i, val in enumerate(S_star)}
            local_S_star = [index_map[i] for i in U if i in index_map]
            K_S = self._group_search(local_S_star, X_U[:, S_star])
            K.extend([[S_star[i] for i in group] for group in K_S])

            index_map = {val: i for i, val in enumerate(U_S_star)}
            local_U_S = [index_map[i] for i in U if i in index_map]
            K_U_S = self._group_search(local_U_S, rS_star)
            K.extend([[U_S_star[i] for i in group] for group in K_U_S])
        else:
            K.append(U)

        return K

    def _estimate_adjacency_matrix(self, X):
        """Estimate adjacency matrix by causal order.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        B = np.zeros((X.shape[1], X.shape[1]), dtype="float64")
        for i in range(1, len(self._causal_order)):
            target_group = self._causal_order[i]
            predictor_groups = self._causal_order[:i]
            predictors = [var for group in predictor_groups for var in group]

            for target in target_group:
                B[target, predictors] = predict_adaptive_lasso(X, predictors, target)

        for target_group in self._causal_order:
            if len(target_group) > 1:
                for k, l in list(itertools.combinations(target_group, 2)):
                    B[k][l] = np.nan
                    B[l][k] = np.nan

        self._adjacency_matrix = B
        return self

    @property
    def adjacency_matrix_(self):
        """Estimated adjacency matrix.

        Returns
        -------
        adjacency_matrix_ : array-like, shape (n_features, n_features)
            The adjacency matrix B of fitted model, where
            n_features is the number of features.
        """
        return self._adjacency_matrix

    @property
    def causal_order_(self):
        """Estimated causal ordering.

        Returns
        -------
        causal_order_ : array-like, shape (n_features)
            The causal order of fitted model, where
            n_features is the number of features.
        """
        return self._causal_order

    def bootstrap(self, X, n_sampling):
        """Evaluate the statistical reliability of DAG based on the bootstrapping.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.
        n_sampling : int
            Number of bootstrapping samples.

        Returns
        -------
        result : BootstrapResult
            Returns the result of bootstrapping.
        """
        # Check parameters
        X = check_array(X)

        if isinstance(n_sampling, (numbers.Integral, np.integer)):
            if not 0 < n_sampling:
                raise ValueError("n_sampling must be an integer greater than 0.")
        else:
            raise ValueError("n_sampling must be an integer greater than 0.")

        # Bootstrapping
        adjacency_matrices = np.zeros([n_sampling, X.shape[1], X.shape[1]])
        total_effects = np.zeros([n_sampling, X.shape[1], X.shape[1]])
        index = np.arange(X.shape[0])
        resampled_indices = []
        for i in range(n_sampling):
            resampled_X, resampled_index = resample(X, index)
            self.fit(resampled_X)
            adjacency_matrices[i] = self._adjacency_matrix

            # Calculate total effects
            for c, from_group in enumerate(self._causal_order):
                for to_group in self._causal_order[c + 1 :]:
                    for from_ in from_group:
                        for to in to_group:
                            total_effects[i, to, from_] = calculate_total_effect(
                                self._adjacency_matrix, from_, to
                            )

            resampled_indices.append(resampled_index)

        return BootstrapResult(adjacency_matrices, total_effects, resampled_indices=resampled_indices)
