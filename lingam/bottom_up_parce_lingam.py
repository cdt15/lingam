"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/site/sshimizu06/lingam
"""

import numbers
import warnings

import numpy as np
from sklearn.utils import check_array, resample
from sklearn.linear_model import LinearRegression
from scipy.stats import gamma
from scipy.stats.distributions import chi2
from .utils import predict_adaptive_lasso

from .bootstrap import BootstrapResult

class BottomUpParceLiNGAM():
    """Implementation of ParceLiNGAM Algorithm [1]_

    References
    ----------
    .. [1] T. Tashiro, S. Shimizu, and A. Hyvärinen.
       ParceLiNGAM: a causal ordering method robust against latent confounders.
       Neural computation, 26.1: 57-83, 2014.
    """
    def __init__(self, random_state=None, alpha=0.05):
        """Construct a BottomUpParceLiNGAM model.

        Parameters
        ----------
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.

        alpha : float, optional (default=0.05)
            Significant level of statistical test.
        """
        self._random_state = random_state
        self._alpha = alpha
        self._causal_order = None
        self._adjacency_matrix = None

        if not 0 < self._alpha:
            raise ValueError('alpha must be an float greater than 0.')

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
        self._causal_order = None
        self._adjacency_matrices = None

        # Check parameters
        X = check_array(X)
        n_features = X.shape[1]

        # Center variables for each group
        X = X - np.tile(np.mean(X, axis=0), (X.shape[0], 1))

        # Search causal orders one by one from the bottom upward
        U = np.arange(n_features)
        # bonferroni correction
        thresh_p = self._alpha / (n_features - 1)

        K_bttm = self._search_causal_order(X, U, thresh_p)

        U_res = list(np.setdiff1d(U, K_bttm))
        K = []
        # Add a list of features whose order is unknown.
        if len(U_res) > 1:
            K = [U_res]
        for i in K_bttm:
            K.append(i)

        self._causal_order = K
        return self._estimate_adjacency_matrix(X)

    def _set_kernel_size(self, X):
        """Set kernel size to median distance between points.
        Use at most 100 points (since median is only a heuristic, and 100 points is sufficient for a robust estimate)."""
        n_samples = X.shape[0]
        if n_samples > 100:
            X_med = X[:100, :]
            n_samples = 100
        else:
            X_med = X

        G = np.sum(X_med * X_med, 1).reshape(n_samples, 1)
        Q = np.tile(G, (1, n_samples))
        R = np.tile(G.T, (n_samples, 1))

        dists = Q + R - 2 * np.dot(X_med, X_med.T)
        dists = dists - np.tril(dists)
        dists = dists.reshape(n_samples ** 2, 1)

        return np.sqrt(0.5 * np.median(dists[dists > 0]))

    def _rbf_dot(self, X, Y, kernel_size):
        """Compute the inner product of radial basis functions."""
        n_samples_X = X.shape[0]
        n_samples_Y = Y.shape[0]

        G = np.sum(X * X, 1).reshape(n_samples_X, 1)
        H = np.sum(Y * Y, 1).reshape(n_samples_Y, 1)
        Q = np.tile(G, (1, n_samples_Y))
        R = np.tile(H.T, (n_samples_X, 1))
        H = Q + R - 2 * np.dot(X, Y.T)

        return np.exp(-H / 2 / (kernel_size ** 2))

    def _hsic_test_gamma(self, X, Y):
        """Perform an HSIC test using gamma approximation."""
        # Set kernel size to median distance between points
        sigx = self._set_kernel_size(X)
        sigy = self._set_kernel_size(Y)

        K = self._rbf_dot(X, X, sigx)
        L = self._rbf_dot(Y, Y, sigy)

        m = X.shape[0]
        bone = np.ones((m, 1))
        H = np.eye(m) - 1 / m * np.ones((m, m))

        # these are slightly biased estimates of centred Gram matrices
        Kc = np.dot(np.dot(H, K), H)
        Lc = np.dot(np.dot(H, L), H)

        # test statistic m*HSICb under H1
        test_stat = 1 / m * np.sum(np.sum(Kc.T * Lc))

        var = (1 / 6 * Kc * Lc) ** 2
        # second subtracted term is bias correction
        var = 1 / m / (m - 1) * (np.sum(np.sum(var)) - np.sum(np.diag(var)))
        # variance under H0
        var = 72 * (m - 4) * (m - 5) / m / (m - 1) / (m - 2) / (m - 3) * var

        K = K - np.diag(np.diag(K))
        L = L - np.diag(np.diag(L))
        mu_X = 1 / m / (m - 1) * np.dot(bone.T, np.dot(K, bone))
        mu_Y = 1 / m / (m - 1) * np.dot(bone.T, np.dot(L, bone))
        # mean under H0
        mean  = 1 / m * (1 + mu_X * mu_Y - mu_X - mu_Y)

        alpha = mean ** 2 / var
        # threshold for hsicArr*m
        beta = np.dot(var, m) / mean
        p = 1 - gamma.cdf(test_stat, alpha, scale=beta)[0][0]

        return test_stat, p

    def _search_causal_order(self, X, U, thresh_p):
        """Search causal orders one by one from the bottom upward."""
        K_bttm = []
        is_search_causal_order = True

        if len(U) <= 1:
            K_bttm = np.append(U, K_bttm).astype(np.int64)
            is_search_causal_order = False

        while is_search_causal_order:
            # Find the most sink variable
            m, exo_vec, reject = self._find_exo_vec(X, U, thresh_p)

            # Conduct statistical test by the p-value or the statistic
            # If statistical test is not rejected
            if not reject:
                # Add index of the exogenous variable to K_bttm
                K_bttm = np.append(m, K_bttm).astype(np.int64)
                U = exo_vec

                # If there is only one candidate for sink variable, the search ends
                if len(U) <= 1:
                    K_bttm = np.append(U, K_bttm).astype(np.int64)
                    is_search_causal_order = False
            # If statistical test is rejected
            else:
                is_search_causal_order = False

        return K_bttm

    def _find_exo_vec(self, X, U, thresh_p):
        """Find the most exogenous vector."""
        max_p = -np.inf
        max_p_stat = np.inf

        exo_vec = []
        cov = np.cov(X.T)
        for j in range(len(U)):
            xi_index = np.setdiff1d(U, U[j])
            xj_index = np.array([U[j]])

            # Compute residuals of least square regressions
            coef = np.dot(np.linalg.pinv(cov[np.ix_(xi_index, xi_index)]), cov[np.ix_(xj_index, xi_index)].reshape(xi_index.shape[0], 1))
            R = X[:, xj_index] - np.dot(X[:, xi_index], coef)

            # HSIC test with Fisher's method
            fisher_p, fisher_stat, reject_temp = self._fisher_hsic_test(X[:, xi_index], R, max_p_stat, thresh_p)

            # Update output
            if fisher_stat < max_p_stat or fisher_p > max_p:
                exo_vec = xi_index
                reject = reject_temp
                max_p = fisher_p
                max_p_stat = fisher_stat

        m = np.setdiff1d(U, exo_vec)
        return m, exo_vec, reject

    def _fisher_hsic_test(self, X, R, max_p_stat, thresh_p):
        """Conduct statistical test by HSIC and Fisher's method."""
        reject = False
        fisher_stat = 0
        n_features = X.shape[1]

        if n_features == 1:
            fisher_stat, fisher_p = self._hsic_test_gamma(X, R)
        else:
            for i in range(n_features):
                _, hsic_p = self._hsic_test_gamma(X[:,[i]], R)
                fisher_stat += np.inf if hsic_p == 0 else - 2 * np.log(hsic_p)

                if fisher_stat > max_p_stat:
                    break
            fisher_p = 1 - chi2.cdf(fisher_stat, df=2 * n_features)

        if fisher_p < thresh_p:
            reject = True

        return fisher_p, fisher_stat, reject

    def _flatten(self, arr):
        """Return a copy of an array flattened in one dimension."""
        return [val for item in arr for val in (self._flatten(item) if hasattr(item, '__iter__') and not isinstance(item, str) else [item])]

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
            # Flatten the array into one dimension
            predictors = self._flatten(self._causal_order[:i])
            coef = predict_adaptive_lasso(X, predictors, self._causal_order[i])
            B[self._causal_order[i], predictors] = coef

        # Set np.nan if order is unknown
        for unk_order in self._causal_order:
            if hasattr(unk_order, '__iter__') and not isinstance(unk_order, str):
                for i in range(len(unk_order)-1):
                    xi = unk_order[i]
                    for xj in unk_order[i+1:]:
                        B[xi, xj] = np.nan
                        B[xj, xi] = np.nan

        self._adjacency_matrix = B
        return self

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
        for i, order in enumerate(self._causal_order):
            if hasattr(order, '__iter__') and from_index in order:
                from_order = i
                break
            elif not hasattr(order, '__iter__') and int(from_index) == int(order):
                from_order = i
                break

        for i, order in enumerate(self._causal_order):
            if hasattr(order, '__iter__') and to_index in order:
                to_order = i
                break
            elif not hasattr(order, '__iter__') and int(to_index) == int(order):
                to_order = i
                break

        if from_order > to_order:
            warnings.warn(f'The estimated causal effect may be incorrect because '
                          f'the causal order of the destination variable (to_index={to_index}) '
                          f'is earlier than the source variable (from_index={from_index}).')

        # from_index + parents indices
        parents = np.where(np.abs(np.nan_to_num(self._adjacency_matrix[from_index])) > 0)[0]
        predictors = [from_index]
        predictors.extend(parents)

        # Estimate total effect
        lr = LinearRegression()
        lr.fit(X[:, predictors], X[:, to_index])

        return lr.coef_[0]

    @property
    def causal_order_(self):
        """Estimated causal ordering.

        Returns
        -------
        causal_order_ : array-like, shape (n_features)
            The causal order of fitted model, where
            n_features is the number of features.
            Set the features as a list if order is unknown.
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
            Set np.nan if order is unknown.
        """
        return self._adjacency_matrix

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
                raise ValueError(
                    'n_sampling must be an integer greater than 0.')
        else:
            raise ValueError('n_sampling must be an integer greater than 0.')

        # Bootstrapping
        adjacency_matrices = np.zeros([n_sampling, X.shape[1], X.shape[1]])
        total_effects = np.zeros([n_sampling, X.shape[1], X.shape[1]])
        for i in range(n_sampling):
            self.fit(resample(X))
            adjacency_matrices[i] = self._adjacency_matrix

            # Calculate total effects
            for c, from_ in enumerate(self._causal_order):
                for to in self._causal_order[c+1:]:
                    if hasattr(from_, '__iter__'):
                        for from_item in from_:
                            total_effects[i, to, from_item] = self.estimate_total_effect(
                                X, from_item, to)
                    else:
                        total_effects[i, to, from_] = self.estimate_total_effect(
                            X, from_, to)

        return BootstrapResult(adjacency_matrices, total_effects)
