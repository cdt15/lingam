"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/site/sshimizu06/lingam
"""
import numbers

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array, resample

from .bootstrap import LongitudinalBootstrapResult
from .direct_lingam import DirectLiNGAM
from .utils import predict_adaptive_lasso


class LongitudinalLiNGAM():
    """Implementation of Longitudinal LiNGAM algorithm [1]_

    References
    ----------
    .. [1] K. Kadowaki, S. Shimizu, and T. Washio. Estimation of causal structures in longitudinal data using non-Gaussianity. In Proc. 23rd IEEE International Workshop on Machine Learning for Signal Processing (MLSP2013), pp. 1--6, Southampton, United Kingdom, 2013.
    """

    def __init__(self, n_lags=1, measure='pwling', random_state=None):
        """Construct a model.

        Parameters
        ----------
        n_lags : int, optional (default=1)
            Number of lags.
        measure : {'pwling', 'kernel'}, default='pwling'
            Measure to evaluate independence : 'pwling' or 'kernel'.
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        """
        self._n_lags = n_lags
        self._measure = measure
        self._random_state = random_state
        self._causal_orders = None
        self._adjacency_matrices = None

    def fit(self, X_list):
        """Fit the model to datasets.

        Parameters
        ----------
        X_list : list, shape [X, ...]
            Longitudinal multiple datasets for training, where ``X`` is an dataset.
            The shape of ``X`` is (n_samples, n_features), 
            where ``n_samples`` is the number of samples and ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Check parameters
        if not isinstance(X_list, (list, np.ndarray)):
            raise ValueError('X_list must be a array-like.')

        if len(X_list) < 2:
            raise ValueError('X_list must be a list containing at least two items')

        self._T = len(X_list)
        self._n = check_array(X_list[0]).shape[0]
        self._p = check_array(X_list[0]).shape[1]
        X_t = []
        for X in X_list:
            X = check_array(X)
            if X.shape != (self._n, self._p):
                raise ValueError('X_list must be a list with the same shape')
            X_t.append(X.T)

        M_tau, N_t = self._compute_residuals(X_t)
        B_t, causal_orders = self._estimate_instantaneous_effects(N_t)
        B_tau = self._estimate_lagged_effects(B_t, M_tau)

        # output B(t,t), B(t,t-τ)
        self._adjacency_matrices = np.empty((self._T, 1+self._n_lags, self._p, self._p))
        self._adjacency_matrices[:, :] = np.nan
        for t in range(1, self._T):
            self._adjacency_matrices[t, 0] = B_t[t]
            for l in range(self._n_lags):
                if t-l == 0:
                    continue
                self._adjacency_matrices[t, l+1] = B_tau[t, l]

        self._causal_orders = causal_orders
        return self

    def bootstrap(self, X_list, n_sampling, start_from_t=1):
        """Evaluate the statistical reliability of DAG based on the bootstrapping.

        Parameters
        ----------
        X_list : array-like, shape (X, ...)
            Longitudinal multiple datasets for training, where ``X`` is an dataset.
            The shape of ''X'' is (n_samples, n_features), 
            where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
        n_sampling : int
            Number of bootstrapping samples.

        Returns
        -------
        results : array-like, shape (BootstrapResult, ...)
            Returns the results of bootstrapping for multiple datasets.
        """
        # Check parameters
        if not isinstance(X_list, (list, np.ndarray)):
            raise ValueError('X_list must be a array-like.')

        if len(X_list) < 2:
            raise ValueError('X_list must be a list containing at least two items')

        self._T = len(X_list)
        self._n = check_array(X_list[0]).shape[0]
        self._p = check_array(X_list[0]).shape[1]
        X_t = []
        for X in X_list:
            X = check_array(X)
            if X.shape != (self._n, self._p):
                raise ValueError('X_list must be a list with the same shape')
            X_t.append(X)

        # Bootstrapping
        adjacency_matrices = np.zeros((n_sampling, self._T, 1+self._n_lags, self._p, self._p))
        total_effects = np.zeros((n_sampling, self._T*self._p, self._T*self._p))
        for i in range(n_sampling):
            resampled_X_t = np.empty((self._T, self._n, self._p))
            indices = np.random.randint(0, self._n, size=(self._n,))
            for t in range(self._T):
                resampled_X_t[t] = X_t[t][indices, :]

            self.fit(resampled_X_t)
            adjacency_matrices[i] = self._adjacency_matrices

            # Calculate total effects
            for from_t in range(start_from_t, self._T):
                for c, from_ in enumerate(self._causal_orders[from_t]):
                    to_t = from_t
                    for to in self._causal_orders[from_t][c+1:]:
                        total_effects[i, to_t * self._p + to, from_t * self._p +
                                      from_] = self.estimate_total_effect(X_t, to_t, to, from_t, from_)

                    for to_t in range(from_t+1, self._T):
                        for to in self._causal_orders[to_t]:
                            total_effects[i, to_t * self._p + to, from_t * self._p +
                                          from_] = self.estimate_total_effect(X_t, to_t, to, from_t, from_)

        return LongitudinalBootstrapResult(self._T, adjacency_matrices, total_effects)

    def estimate_total_effect(self, X_t, to_t, to_index, from_t, from_index):
        """Estimate total effect using causal model.

        Parameters
        ----------
        X_t : array-like, shape (n_samples, n_features)
            Original data, where n_samples is the number of samples
            and n_features is the number of features.
        to_t : 
            The timepoint of destination variable.
        to_index : 
            Index of destination variable to estimate total effect.
        from _t : 
            The timepoint of source variable.
        from_index : 
            Index of source variable to estimate total effect.

        Returns
        -------
        total_effect : float
            Estimated total effect.
        """
        # X + lagged X
        # n_features * (to + from + n_lags)
        X_joined = np.zeros((self._n, self._p*(2+self._n_lags)))
        X_joined[:, 0:self._p] = X_t[to_t]
        for tau in range(1+self._n_lags):
            pos = self._p + self._p*tau
            X_joined[:, pos:pos+self._p] = X_t[from_t-tau]

        am = np.concatenate([*self._adjacency_matrices[from_t]], axis=1)

        # from_index + parents indices
        parents = np.where(np.abs(am[from_index]) > 0)[0]
        predictors = [from_index + self._p]
        predictors.extend(parents + self._p)

        # Estimate total effect
        lr = LinearRegression()
        lr.fit(X_joined[:, predictors], X_joined[:, to_index])

        return lr.coef_[0]

    def _compute_residuals(self, X_t):
        """Compute residuals N(t)"""
        M_tau = np.zeros((self._T, self._n_lags, self._p, self._p))
        N_t = np.zeros((self._T, self._p, self._n))

        for t in range(1, self._T):
            # predictors
            X_predictors = np.zeros((self._n, self._p*(1+self._n_lags)))
            for tau in range(self._n_lags):
                pos = self._p * tau
                X_predictors[:, pos:pos+self._p] = X_t[t-(tau+1)].T

            # estimate M(t,t-τ) by regression
            X_target = X_t[t].T
            for i in range(self._p):
                reg = LinearRegression()
                reg.fit(X_predictors, X_target[:, i])
                for tau in range(self._n_lags):
                    pos = self._p * tau
                    M_tau[t, tau, i] = reg.coef_[pos:pos+self._p]

            # Compute N(t)
            N_t[t] = X_t[t]
            for tau in range(self._n_lags):
                N_t[t] = N_t[t] - np.dot(M_tau[t, tau], X_t[t-(tau+1)])

        return M_tau, N_t

    def _estimate_instantaneous_effects(self, N_t):
        """Estimate instantaneous effects B(t,t) by applying LiNGAM"""
        causal_orders = [[np.nan]*self._p]
        B_t = np.zeros((self._T, self._p, self._p))
        for t in range(1, self._T):
            model = DirectLiNGAM(measure=self._measure)
            model.fit(N_t[t].T)
            causal_orders.append(model.causal_order_)
            B_t[t] = model.adjacency_matrix_
        return B_t, causal_orders

    def _estimate_lagged_effects(self, B_t, M_tau):
        """Estimate lagged effects B(t,t-τ)"""
        B_tau = np.zeros((self._T, self._n_lags, self._p, self._p))
        for t in range(self._T):
            for tau in range(self._n_lags):
                B_tau[t, tau] = np.dot(np.eye(self._p) - B_t[t], M_tau[t, tau])
        return B_tau

    @property
    def causal_orders_(self):
        """Estimated causal ordering.

        Returns
        -------
        causal_order_ : array-like, shape (causal_order, ...)
            The causal order of fitted models for B(t,t).
            The shape of causal_order is (n_features), 
            where ``n_features`` is the number of features.
        """
        return self._causal_orders

    @property
    def adjacency_matrices_(self):
        """Estimated adjacency matrices.

        Returns
        -------
        adjacency_matrices_ : array-like, shape ((B(t,t), B(t,t-1), ..., B(t,t-τ)), ...)
            The list of adjacency matrix B(t,t) and B(t,t-τ) for longitudinal datasets.
            The shape of B(t,t) and B(t,t-τ) is (n_features, n_features), where 
            ``n_features`` is the number of features.
            **If the previous data required for the calculation are not available, 
            such as B(t,t) or B(t,t-τ) at t=0, all elements of the matrix are nan**.
        """
        return self._adjacency_matrices
