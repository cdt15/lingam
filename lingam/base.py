"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/site/sshimizu06/lingam
"""

import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array

from .bootstrap import BootstrapMixin
from .utils import predict_adaptive_lasso


class _BaseLiNGAM(BootstrapMixin, metaclass=ABCMeta):
    """Base class for all LiNGAM algorithms."""

    def __init__(self, random_state=None):
        """Construct a _BaseLiNGAM model.

        Parameters
        ----------
        random_state : int, optional (default=None)
            random_state is the seed used by the random number generator.
        """
        self._random_state = random_state
        self._causal_order = None
        self._adjacency_matrix = None

    @abstractmethod
    def fit(self, X):
        """Subclasses should implement this method!
        Fit the model to X.

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

    def estimate_total_effect(self, X, from_index, to_index):
        """Estimate total effect using causal model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Original data, where n_samples is the number of samples
            and n_features is the number of features.
        from_index : 
            Index of source variable to estimate total effect.
        to_index : 
            Index of destination variable to estimate total effect.

        Returns
        -------
        total_effect : float
            Estimated total effect.
        """
        # Check parameters
        X = check_array(X)

        # Check from/to causal order
        from_order = self._causal_order.index(from_index)
        to_order = self._causal_order.index(to_index)
        if from_order > to_order:
            warnings.warn(f'The estimated causal effect may be incorrect because ' 
                          f'the causal order of the destination variable (to_index={to_index}) '
                          f'is earlier than the source variable (from_index={from_index}).')

        # from_index + parents indices
        parents = np.where(np.abs(self._adjacency_matrix[from_index]) > 0)[0]
        predictors = [from_index]
        predictors.extend(parents)

        # Estimate total effect
        lr = LinearRegression()
        lr.fit(X[:, predictors], X[:, to_index])

        return lr.coef_[0]

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
        B = np.zeros([X.shape[1], X.shape[1]], dtype='float64')
        for i in range(1, len(self._causal_order)):
            B[self._causal_order[i], self._causal_order[:i]] = predict_adaptive_lasso(
                X, self._causal_order[:i], self._causal_order[i])

        self._adjacency_matrix = B
        return self

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
