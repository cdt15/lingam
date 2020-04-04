"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/site/sshimizu06/lingam
"""

import numpy as np
from sklearn.utils import check_array, resample
from sklearn.linear_model import LassoLarsIC, LinearRegression
from statsmodels.tsa.vector_ar.var_model import VAR

from lingam import DirectLiNGAM

from .base import _BaseLiNGAM
from .bootstrap import BootstrapResult


class VARLiNGAM:
    """Implementation of VAR-LiNGAM Algorithm [1]_

    References
    ----------
    .. [1] Aapo Hyvärinen, Kun Zhang, Shohei Shimizu, Patrik O. Hoyer.
       Estimation of a Structural Vector Autoregression Model Using Non-Gaussianity. Journal of Machine Learning Research, 11: 1709-1731, 2010.
    """
    
    def __init__(self, lags=1, criterion='bic', prune=False, ar_coefs=None, lingam_model=None, random_state=None):
        """Construct a VARLiNGAM model.

        Parameters
        ----------
        lags : int, optional (default=1)
            Number of lags.
        criterion : {‘aic’, ‘fpe’, ‘hqic’, ‘bic’, None}, optional (default='bic')
            Criterion to decide the best lags within ``lags``.
            Searching the best lags is disabled if ``criterion`` is ``None``.
        prune : boolean, optional (default=False)
            Whether to prune the adjacency matrix or not.
        ar_coefs : array-like, optional (default=None)
            Coefficients of AR model. Estimating AR model is skipped if specified ``ar_coefs``.
            Shape must be (``lags``, n_features, n_features).
        lingam_model : constructor
            Constructor of a LiNGAM algorithm which inherits _BaseLiNGAM.
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        """
        self._lags = lags
        self._criterion = criterion
        self._prune = prune
        self._ar_coefs = check_array(ar_coefs, allow_nd=True) if ar_coefs is not None else None
        self._lingam_model = lingam_model
        self._random_state = random_state

    def fit(self, X):
        """Fit the model to X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.

        returns
        -------
        self : object
            Returns the instance itself.
        """
        self._causal_order = None
        self._adjacency_matrices = None

        X = check_array(X)

        lingam_model = self._lingam_model
        if lingam_model is None:
            lingam_model = DirectLiNGAM
        elif not issubclass(lingam_model, _BaseLiNGAM):
            raise ValueError('lingam_model must be a subclass of _BaseLiNGAM')

        M_taus = self._ar_coefs

        if M_taus is None:
            M_taus, lags, residuals = self._estimate_var_coefs(X)
        else:
            lags = M_taus.shape[0]
            residuals = self._calc_residuals(X, M_taus, lags)

        model = DirectLiNGAM()
        model.fit(residuals)

        B_taus = self._calc_b(X, model.adjacency_matrix_, M_taus)

        if self._prune:
            B_taus = self._pruning(X, B_taus, model.causal_order_)

        self._ar_coefs = M_taus
        self._lags = lags
        self._residuals = residuals

        self._causal_order = model.causal_order_
        self._adjacency_matrices = B_taus

        return self
        
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
        X = check_array(X)

        n_samples = X.shape[0]
        n_features = X.shape[1]
        lags = self._lags

        self.fit(X)

        residuals = self._residuals
        ar_coefs = self._ar_coefs
        
        adjacency_matrices = []
        for i in range(n_sampling):
            sampled_residuals = resample(residuals, n_samples=n_samples)

            resampled_X = np.zeros((n_samples, n_features))
            for j in range(n_samples):
                if j < lags:
                    resampled_X[j, :] = sampled_residuals[j]
                    continue

                ar = np.zeros((1, n_features))
                for t, M in enumerate(ar_coefs):
                    ar += np.dot(M, X[j - t - 1, :].T).T

                resampled_X[j, :] = ar + sampled_residuals[j]
            
            self.fit(resampled_X)
            adjacency_matrices.append(self._adjacency_matrices)

        cated_adj_matrix = []
        for matrix in adjacency_matrices:
            m = np.concatenate([*matrix], axis=1)
            cated_adj_matrix.append(m)

        return BootstrapResult(cated_adj_matrix)

    def _estimate_var_coefs(self, X):
        """Estimate coefficients of VAR"""
        var = VAR(X)
        result = var.fit(maxlags=self._lags, ic=self._criterion, trend='nc')

        return result.coefs, result.k_ar, result.resid

    def _calc_residuals(self, X, M_taus, lags):
        """Calculate residulas"""
        X = X.T
        n_features = X.shape[0]
        n_samples = X.shape[1]

        residuals = np.zeros((n_features, n_samples))
        for t in range(n_samples):
            if t - lags < 0:
                continue

            estimated = np.zeros((X.shape[0], 1))
            for tau in range(1, lags + 1):
                estimated += np.dot(M_taus[tau - 1], X[:, t - tau].reshape((-1, 1)))

            residuals[:, t] = X[:, t] - estimated.reshape((-1,))

        residuals = residuals[:, lags:].T

        return residuals

    def _calc_b(self, X, B0, M_taus):
        """Calculate B"""
        n_features = X.shape[1]
        
        B_taus = np.array([B0])
        
        for M in M_taus:
            B_t = np.dot((np.eye(n_features) - B0), M)
            B_taus = np.append(B_taus, [B_t], axis=0)

        return B_taus

    def _pruning(self, X, B_taus, causal_order):
        """Prune edges"""
        n_features = X.shape[1]

        stacked = [np.flip(X, axis=0)]
        for i in range(self._lags):
            stacked.append(np.roll(stacked[-1], -1, axis=0))
        blocks = np.array(list(zip(*stacked)))[:-self._lags]

        for i in range(n_features):
            causal_order_no = causal_order.index(i)
            ancestor_indexes = causal_order[:causal_order_no]
            
            obj = np.zeros((len(blocks)))
            exp = np.zeros((len(blocks), causal_order_no + n_features * self._lags))
            for j, block in enumerate(blocks):
                obj[j] = block[0][i]
                exp[j :] = np.concatenate([block[0][ancestor_indexes].flatten(), block[1:][:].flatten()], axis=0)
            
            # adaptive lasso
            gamma = 1.0
            lr = LinearRegression()
            lr.fit(exp, obj)
            weight = np.power(np.abs(lr.coef_), gamma)
            reg = LassoLarsIC(criterion='bic')
            reg.fit(exp * weight, obj)
            coef = reg.coef_ * weight
            
            B_taus[0][i, ancestor_indexes] = coef[:causal_order_no]
            for j in range(len(B_taus[1:])):
                B_taus[j + 1][i, :] = coef[causal_order_no + n_features * j: causal_order_no + n_features * j + n_features]

        return B_taus

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
    def adjacency_matrices_(self):
        """Estimated adjacency matrix.

        Returns
        -------
        adjacency_matrices_ : array-like, shape (lags, n_features, n_features)
            The adjacency matrix of fitted model, where 
            n_features is the number of features.
        """
        return self._adjacency_matrices
