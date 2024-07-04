"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""

import itertools
import warnings
import numpy as np
from sklearn.utils import check_array, check_scalar
from sklearn.linear_model import LassoLarsCV, LinearRegression
from sklearn.preprocessing import StandardScaler

from .direct_lingam import DirectLiNGAM


class HighDimDirectLiNGAM(DirectLiNGAM):
    """ An implementation of the high-dimensional LiNGAM algorithm. [1]_

    References
    ----------
    .. [1] Wang, Y. Samuel, and Mathias Drton. "High-dimensional causal discovery under non-Gaussianity."
       Biometrika 107.1 (2020): 41-59.
    """

    def __init__(self, J=3, K=4, alpha=0.5, estimate_adj_mat=True, random_state=None):
        """Construct a HighDimDirectLiNGAM model.

        Parameters
        ----------
        J : int, optional (default=3)
            Assumed largest in-degree.
        K : int, optional (default=4)
            The degree of the moment which is non-Gaussianity.
        alpha : float, optional (default=0.5)
            The value for pruning away false parents.
        estimate_adj_mat : bool, optional (default=True)
            If Fase, skip the estimation of the adjacency matrix.
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        """

        super().__init__(random_state=random_state)

        self._J = check_scalar(J, "J", int, min_val=2, include_boundaries="neither")
        self._K = check_scalar(K, "K", int, min_val=1)
        self._alpha = check_scalar(alpha, "alpha", (int, float), min_val=0, max_val=1)
        self._estimate_adj_mat = check_scalar(estimate_adj_mat, "estimate_adj_mat", bool)

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
            Fitted model.
        """

        self._Y = check_array(X)

        self._yty = self._Y.T @ self._Y

        cut_off = 0

        theta = []
        p = self._Y.shape[1]

        psi = list(range(p))

        prune_stats = np.full((p, p), 1e5)
        np.fill_diagonal(prune_stats, 0)

        while len(psi) > 1:
            new_stats = []
            for v in psi:
                cond_set = set.intersection(set(theta), set(np.argwhere(prune_stats[v] > cut_off).flatten()))
                cond_set = set.union(cond_set, set(theta[-1:] if len(theta) > 0 else set()))

                last_root = theta[-1] if len(theta) > 0 else None
                stats = self._get_prune_stats(v, psi, self._K, last_root, cond_set, self._J)

                new_stats.append(stats)
            new_stats = np.array(new_stats)

            prune_stats[psi, :] = np.min([prune_stats[psi, :], new_stats], axis=0)
            np.fill_diagonal(prune_stats, 0)

            max_taus = np.max(prune_stats[np.ix_(psi, psi)], axis=1)

            r = psi[np.argmin(max_taus)]

            cut_off = max(cut_off, min(max_taus) * self._alpha)

            theta.append(r)
            psi.remove(r)

        self._causal_order = [*theta, *psi]

        if not self._estimate_adj_mat:
            return self

        if self._Y.shape[0] <= self._Y.shape[1]:
            warnings.warn(
                "Since n_samples <= n_features, the adjacency matrix is "
                "estimated with LassoLarsCV instead of LassoLarsIC."
            )
            return self._estimate_adjacency_matrix2(self._Y)

        return self._estimate_adjacency_matrix(self._Y)

    def _calc_tau(self, k, pa, ch):
        return abs(np.mean((pa ** (k - 1)) * ch) * np.mean(pa ** 2) - np.mean(pa ** k) * np.mean(pa * ch))

    def _calc_taus(self, pa, ch, k, cond_sets, an_sets):
        n = self._Y.shape[0]
        ret = np.empty(self._Y.shape[1])
        ret.fill(1e10)

        for z in range(len(cond_sets)):
            cond = cond_sets[z]

            b = np.linalg.pinv(self._yty[np.ix_(cond, cond)]) @ self._yty[cond, pa]
            resid = (self._Y[:, [pa]] - self._Y[:, cond] @ b.reshape(len(cond), 1)).flatten()

            resid_k_1 = np.power(resid, k - 1)
            resid_var = np.mean(np.power(resid, 2))
            resid_k = np.mean(np.power(resid, k))

            for i in range(len(ch)):
                value = (1.0 / n) * np.dot(resid_k_1, self._Y[:, ch[i]]) * resid_var\
                    - resid_k * (1.0 / n) * np.dot(resid, self._Y[:, ch[i]])
                ret[ch[i]] = np.min([ret[ch[i]], abs(value)])

            for i in range(an_sets.shape[1]):
                value = (1.0 / n) * np.dot(resid_k_1.T, self._Y[:, an_sets[z, i]]) * resid_var\
                    - resid_k * (1.0 / n) * np.dot(resid.T, self._Y[:, an_sets[z, i]])
                ret[an_sets[z, i]] = np.min([ret[an_sets[z, i]], abs(value)])

            return ret

    def _get_prune_stats(self, i, j, K, last_root, condition_set, J):
        j = list(set(j) - {i})
        p = self._Y.shape[1]
        prune_stat = np.array([1e5 for _ in range(p)])

        if last_root is None:
            prune_stat[j] = [self._calc_tau(K, self._Y[:, i], self._Y[:, j_]) for j_ in j]
            return prune_stat

        size_of_set = min(J, len(condition_set))

        if len(condition_set - {last_root}) == 1:
            condition_sub_set = np.array([list(condition_set - {last_root})]).reshape(1, 1)
        else:
            condition_sub_set = list(itertools.combinations(list(condition_set - {last_root}), size_of_set - 1))
            condition_sub_set = np.array([list(e) for e in condition_sub_set])

        an_sets = np.array([list(condition_set - set(x)) for x in condition_sub_set])
        if len(an_sets) == 1 and an_sets.shape[1] > 0:
            an_sets = an_sets.T

        condition_sub_set = np.concatenate([
            np.full(len(condition_sub_set), last_root).reshape(-1, 1),
            condition_sub_set
        ], axis=1).astype(int)

        prune_stat = self._calc_taus(i, j, K, condition_sub_set, an_sets)
        return prune_stat

    def _estimate_adjacency_matrix2(self, X):
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

        B = np.zeros([X.shape[1], X.shape[1]], dtype="float64")
        for i in range(1, len(self._causal_order)):
            target = self._causal_order[i]
            predictors = self._causal_order[:i]

            # target is exogenous variables if predictors are empty
            if len(predictors) == 0:
                continue

            B[target, predictors] = self._predict_adaptive_lasso(X, predictors, target)

        self._adjacency_matrix = B
        return self

    def _predict_adaptive_lasso(self, X, predictors, target, gamma=1.0):
        """Predict with Adaptive Lasso.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        predictors : array-like, shape (n_predictors)
            Indices of predictor variable.
        target : int
            Index of target variable.

        Returns
        -------
        coef : array-like, shape (n_features)
            Coefficients of predictor variable.
        """
        # Standardize X
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        # Pruning with Adaptive Lasso
        lr = LinearRegression()
        lr.fit(X_std[:, predictors], X_std[:, target])
        weight = np.power(np.abs(lr.coef_), gamma)
        reg = LassoLarsCV()
        reg.fit(X_std[:, predictors] * weight, X_std[:, target])
        pruned_idx = np.abs(reg.coef_ * weight) > 0.0

        # Calculate coefficients of the original scale
        coef = np.zeros(reg.coef_.shape)
        if pruned_idx.sum() > 0:
            lr = LinearRegression()
            pred = np.array(predictors)
            lr.fit(X[:, pred[pruned_idx]], X[:, target])
            coef[pruned_idx] = lr.coef_

        return coef
