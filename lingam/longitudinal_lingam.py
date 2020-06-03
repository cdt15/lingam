"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/site/sshimizu06/lingam
"""
import numbers
import numpy as np
from sklearn.utils import check_array, resample
from sklearn.linear_model import LinearRegression

from .direct_lingam import DirectLiNGAM
from .bootstrap import LongitudinalBootstrapResult


class LongitudinalLiNGAM():
    """Implementation of Longitudinal LiNGAM algorithm [1]_

    References
    ----------
    .. [1] K. Kadowaki, S. Shimizu, and T. Washio. Estimation of causal structures in longitudinal data using non-Gaussianity. In Proc. 23rd IEEE International Workshop on Machine Learning for Signal Processing (MLSP2013), pp. 1--6, Southampton, United Kingdom, 2013.
    """

    def __init__(self, random_state=None, n_lags=1):
        """Construct a model.

        Parameters
        ----------
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        n_lags : int, optional (default=1)
            Number of lags.
        """
        self._random_state = random_state
        self._n_lags = n_lags
        self._causal_orders = None
        self._adjacency_matrices = None

    def fit(self, X_list):
        """Fit the model to datasets.

        Parameters
        ----------
        X_list : list, shape [X, ...]
            Longitudinal multiple datasets for training, where ``X`` is an dataset.
            The shape of ''X'' is (n_samples, n_features), 
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

        n_samples = check_array(X_list[0]).shape[0]
        n_features = check_array(X_list[0]).shape[1]
        X_t = []
        for X in X_list:
            X = check_array(X)
            if X.shape != (n_samples, n_features):
                raise ValueError('X_list must be a list with the same shape')
            X_t.append(X.T)

        self._T = len(X_t)  # Number of time points
        self._n = n_samples  # Number of samples
        self._p = n_features  # Number of features

        M_tau, N_t = self._compute_residuals(X_t)
        B_t, causal_orders = self._estimate_instantaneous_effects(N_t)
        B_tau = self._estimate_lagged_effects(B_t, M_tau)

        # output B(t,t), B(t,t-τ)
        self._adjacency_matrices = np.zeros((self._T, 1+self._n_lags, self._p, self._p))
        for t in range(self._T):
            self._adjacency_matrices[t, 0] = B_t[t]
            for l in range(self._n_lags):
                self._adjacency_matrices[t, l+1] = B_tau[t, l]
        self._causal_orders = causal_orders
        return self

    def bootstrap(self, X_list, n_sampling):
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

        n_samples = check_array(X_list[0]).shape[0]
        n_features = check_array(X_list[0]).shape[1]
        X_t = []
        for X in X_list:
            X = check_array(X)
            if X.shape != (n_samples, n_features):
                raise ValueError('X_list must be a list with the same shape')
            X_t.append(X)

        self._T = len(X_t)  # Number of time points
        self._n = n_samples  # Number of samples
        self._p = n_features  # Number of features

        adjacency_matrices = np.zeros(
            (n_sampling, self._T, 1+self._n_lags, self._p, self._p))
        for i in range(n_sampling):
            print('sampling:', i)
            resampled_X_t = np.empty((self._T, self._n, self._p))
            indices = np.random.randint(0, self._n, size=(self._n,))
            for t in range(self._T):
                resampled_X_t[t] = X_t[t][indices, :]

            model = self.fit(resampled_X_t)
            adjacency_matrices[i] = model.adjacency_matrices_
        return LongitudinalBootstrapResult(adjacency_matrices, self._T)

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
        causal_orders = []
        B_t = np.empty((self._T, self._p, self._p))
        for t in range(self._T):
            model = DirectLiNGAM()
            model.fit(N_t[t].T)
            causal_orders.append(model.causal_order_)
            B_t[t] = model.adjacency_matrix_
        return B_t, causal_orders

    def _estimate_lagged_effects(self, B_t, M_tau):
        """Estimate lagged effects B(t,t-τ)"""
        B_tau = np.empty((self._T, self._n_lags, self._p, self._p))
        for t in range(self._T):
            for tau in range(self._n_lags):
                B_tau[t, tau] = np.dot(np.eye(self._p) - B_t[t], M_tau[t, tau])
        return B_tau

    @property
    def causal_orders_(self):
        """Estimated causal ordering.

        Returns
        -------
        causal_order_ : array-like, shape (n_features)
            The causal order of fitted model, where 
            n_features is the number of features.
        """
        return self._causal_orders

    @property
    def adjacency_matrices_(self):
        """Estimated adjacency matrices.

        Returns
        -------
        adjacency_matrices_ : array-like, shape (B, ...)
            The list of adjacency matrix B for multiple datasets.
            The shape of B is (n_features, n_features), where 
            n_features is the number of features.
        """
        return self._adjacency_matrices
