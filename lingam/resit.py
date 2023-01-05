"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""

import numpy as np
from sklearn.utils import check_array

from .base import _BaseLiNGAM
from .hsic import hsic_test_gamma


class RESIT(_BaseLiNGAM):
    """Implementation of RESIT(regression with subsequent independence test) Algorithm [1]_

    References
    ----------
    .. [1] Jonas Peters, Joris M Mooij, Dominik Janzing, and Bernhard Sch ̈olkopf.
       Causal discovery with continuous additive noise models.
       Journal of Machine Learning Research, 15:2009-2053, 2014.

    Notes
    -----
    RESIT algorithm returns an **adjacency matrix consisting of zeros or ones**,
    rather than an adjacency matrix consisting of causal coefficients,
    in order to estimate nonlinear causality.
    """

    def __init__(self, regressor, random_state=None, alpha=0.01):
        """Construct a RESIT model.

        Parameters
        ----------
        regressor : regressor object implementing 'fit' and 'predict' function (default=None)
            Regressor to compute residuals.
            This regressor object must have ``fit`` method and ``predict`` function like scikit-learn's model.
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        alpha : float, optional (default=0.01)
            Alpha level for HSIC independence test when removing superfluous edges.
        """
        # Check parameters
        if regressor is None:
            raise ValueError("Specify regression model in 'regressor'.")
        else:
            if not (hasattr(regressor, "fit") and hasattr(regressor, "predict")):
                raise ValueError("'regressor' has no fit or predict method.")

        if alpha < 0.0:
            raise ValueError("alpha must be an float greater than 0.")

        super().__init__(random_state)
        self._alpha = alpha
        self._reg = regressor

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

        # Determine topological order
        pa, pi = self._estimate_order(X)

        # Remove superfluous edges
        pa = self._remove_edges(X, pa, pi)

        # Create adjacency matrix from parent-child relationship
        adjacency_matrix = np.zeros([n_features, n_features])
        for i, parents in pa.items():
            for p in parents:
                adjacency_matrix[i, p] = 1

        self._causal_order = pi
        self._adjacency_matrix = adjacency_matrix
        return self

    def _estimate_order(self, X):
        """Determine topological order"""
        S = np.arange(X.shape[1])
        pa = {}
        pi = []
        for _ in range(X.shape[1]):

            if len(S) == 1:
                pa[S[0]] = []
                pi.insert(0, S[0])
                continue

            hsic_stats = []
            for k in S:
                # Regress Xk on {Xi}
                predictors = [i for i in S if i != k]
                self._reg.fit(X[:, predictors], X[:, k])
                residual = X[:, k] - self._reg.predict(X[:, predictors])
                # Measure dependence between residuals and {Xi}
                hsic_stat, hsic_p = hsic_test_gamma(residual, X[:, predictors])
                hsic_stats.append(hsic_stat)

            k = S[np.argmin(hsic_stats)]
            S = S[S != k]
            pa[k] = S.tolist()
            pi.insert(0, k)

        return pa, pi

    def _remove_edges(self, X, pa, pi):
        """Remove superfluous edges"""
        for k in range(1, X.shape[1]):
            parents = pa[pi[k]].copy()

            for l in parents:
                # Regress Xk on {Xi}
                predictors = [i for i in pa[pi[k]] if i != l]
                # if len(predictors) <= 1:
                if len(predictors) < 1:
                    continue
                self._reg.fit(X[:, predictors], X[:, pi[k]])
                residual = X[:, pi[k]] - self._reg.predict(X[:, predictors])
                # Measure dependence between residuals and {Xi}
                _, hsic_p = hsic_test_gamma(residual, X[:, predictors])
                # eliminate edge
                if hsic_p > self._alpha:
                    pa[pi[k]].remove(l)

        return pa

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
            **Because RESIT is a nonlinear algorithm, it cannot estimate the total effect and always returns a value of zero**
        """
        return 0

    def get_error_independence_p_values(self, X):
        """Calculate the p-value matrix of independence between error variables.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Original data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        independence_p_values : array-like, shape (n_features, n_features)
            **RESIT always returns zero**
        """
        n_features = X.shape[1]
        p_values = np.zeros([n_features, n_features])
        return p_values
