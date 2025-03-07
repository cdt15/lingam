"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""

import copy
import itertools

import numpy as np
from pygam import LinearGAM
from scipy.stats import chi2
from sklearn.utils import check_array

from .hsic import hsic_test_gamma
from .utils import f_correlation


class MultiGroupCAMUV:
    """Implementation of CAM-UV Algorithm with multiple groups [1]_ [2]_ [3]_

    References
    ----------
    .. [1] T.N.Maeda and S.Shimizu. Causal additive models with unobserved variables.
       In Proc. Thirty-Seventh Conference on Uncertainty in Artificial Intelligence (UAI). PMLR  161:97-106, 2021.
    .. [2] T. N. Maeda and S. Shimizu. Use of prior knowledge to discover causal additive models with unobserved
       variables and its application to time series data. Behaviormetrika, xx(xx): 1-19, 2024.
    .. [3] S. Shimizu. Joint estimation of linear non-Gaussian acyclic models. Neurocomputing, 81: 104-107, 2012.
    """

    def __init__(
        self,
        alpha=0.01,
        num_explanatory_vals=2,
        independence="hsic",
        ind_corr=0.5,
        prior_knowledge=None,
    ):
        """Construct a CAM-UV model.

        Parameters
        ----------
        alpha : float, optional (default=0.01)
            Alpha level.
        num_explanatory_vals : int, optional (default=2)
            Maximum number of explanatory variables.
        independence : {'hsic', 'fcorr'}, optional (default='hsic')
            Methods to determine independence. If 'hsic' is set, test for independence by HSIC.
            If 'fcorr' is set, independence is determined by F-correlation.
        ind_corr : float, optional (default=0.5)
            The threshold value for determining independence by F-correlation;
            independence is determined when the value of F-correlation is below this threshold value.
        prior_knowledge : array-like, shape ((index, index), ...), optional (default=None)
            List of variable pairs indicating prior knowledge.
            If (0, 3) is included , it means that X0 cannot be a cause of X3.
        """

        # Check parameters
        if num_explanatory_vals <= 0:
            raise ValueError("num_explanatory_vals must be > 0.")

        if alpha < 0:
            raise ValueError("alpha must be >= 0.")

        if independence not in ("hsic", "fcorr"):
            raise ValueError("independence must be 'hsic' or 'fcorr'.")

        if ind_corr < 0.0:
            raise ValueError("ind_corr must be an float greater than 0.")

        self._num_explanatory_vals = num_explanatory_vals
        self._alpha = alpha
        self._independence = independence
        self._ind_corr = ind_corr
        self._pk_dict = self._make_pk_dict(prior_knowledge)

    def fit(self, X_list):
        """Fit the model to multiple datasets.

        Parameters
        ----------
        X_list : list, shape [X, ...]
            Multiple datasets for training, where ``X`` is an dataset.
            The shape of ''X'' is (n_samples, n_features),
            where ``n_samples`` is the number of samples and ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        X_list = self._check_X_list(X_list)

        d = X_list[0].shape[1]
        N = self._get_neighborhoods(X_list)
        P = self._find_parents(X_list, self._num_explanatory_vals, N)

        U = []
        for i in range(d):
            for j in range(d)[i + 1 :]:
                if (i in P[j]) or (j in P[i]):
                    continue
                if (i not in N[j]) or (j not in N[i]):
                    continue

                in_X_list = []
                in_Y_list = []
                for X in X_list:
                    n = X.shape[0]
                    i_residual = self._get_residual(X, i, P[i])
                    j_residual = self._get_residual(X, j, P[j])
                    in_X_list.append(np.reshape(i_residual, [n, 1]))
                    in_Y_list.append(np.reshape(j_residual, [n, 1]))

                if not self._is_independent(in_X_list, in_Y_list):
                    if not set([i, j]) in U:
                        U.append(set([i, j]))

        self._U = U
        self._P = P

        return self._estimate_adjacency_matrix(X_list[0], P, U)

    def _make_pk_dict(self, prior_knowledge):
        if prior_knowledge is None:
            return None

        pk_dict = dict()
        for pair in prior_knowledge:
            if not pair[1] in pk_dict:
                pk_dict[pair[1]] = [pair[0]]
            else:
                pk_dict[pair[1]].append(pair[0])
        return pk_dict

    def _check_X_list(self, X_list):
        """Check input X list."""
        if not isinstance(X_list, list):
            raise ValueError("X_list must be a list.")

        if len(X_list) < 2:
            raise ValueError("X_list must be a list containing at least two items")

        self._k = len(X_list)
        self._n_features = check_array(X_list[0]).shape[1]
        X_list_ = []
        for X in X_list:
            X_ = check_array(X)
            if X_.shape[1] != self._n_features:
                raise ValueError(
                    "X_list must be a list with the same number of features"
                )
            X_list_.append(X_)

        return X_list_

    def _get_residual(self, X, explained_i, explanatory_ids):
        explanatory_ids = list(explanatory_ids)

        if len(explanatory_ids) == 0:
            residual = X[:, explained_i]
        else:
            gam = LinearGAM().fit(X[:, explanatory_ids], X[:, explained_i])
            residual = X[:, explained_i] - gam.predict(X[:, explanatory_ids])
        return residual

    def _is_independent(self, X_list, Y_list):
        if self._independence == "hsic":
            threshold = self._alpha
        elif self._independence == "fcorr":
            threshold = self._ind_corr
        is_independent, _ = self._is_independent_by(X_list, Y_list, threshold)
        return is_independent

    def _is_independent_by(self, X_list, Y_list, threshold):
        is_independent = False
        value = 0.0
        if self._independence == "hsic":
            fisher_stat = 0
            for X, Y in zip(X_list, Y_list):
                _, hsic_p = hsic_test_gamma(X, Y)
                fisher_stat += np.inf if hsic_p == 0 else -2 * np.log(hsic_p)
            fisher_p = chi2.sf(fisher_stat, df=2 * self._k)
            is_independent = fisher_p > threshold
            value = fisher_p
        elif self._independence == "fcorr":
            max_f_corr = 0.0
            for X, Y in zip(X_list, Y_list):
                f_corr = f_correlation(X, Y)
                if f_corr > max_f_corr:
                    max_f_corr = f_corr
            is_independent = max_f_corr < threshold
            value = max_f_corr
        return is_independent, value

    def _get_neighborhoods(self, X_list):
        d = X_list[0].shape[1]
        N = [set() for i in range(d)]
        for i in range(d):
            for j in range(d)[i + 1 :]:

                in_X_list = []
                in_Y_list = []
                for X in X_list:
                    n = X.shape[0]
                    in_X_list.append(np.reshape(X[:, i], [n, 1]))
                    in_Y_list.append(np.reshape(X[:, j], [n, 1]))

                if not self._is_independent(in_X_list, in_Y_list):
                    N[i].add(j)
                    N[j].add(i)
        return N

    def _find_parents(self, X_list, maxnum_vals, N):
        d = X_list[0].shape[1]
        P = [set() for i in range(d)]  # Parents
        t = 2
        Y_list = [copy.deepcopy(X) for X in X_list]

        while True:
            changed = False
            variables_set_list = list(itertools.combinations(set(range(d)), t))
            for variables_set in variables_set_list:
                variables_set = set(variables_set)

                if not self._check_identified_causality(variables_set, P):
                    continue

                child, is_independence_with_K = self._get_child(
                    X_list, variables_set, P, N, Y_list
                )
                if child is None:
                    continue
                if not is_independence_with_K:
                    continue

                parents = variables_set - {child}
                if not self._check_independence_withou_K(parents, child, P, N, Y_list):
                    continue

                for parent in parents:
                    P[child].add(parent)
                    changed = True
                    Y_list_ = []
                    for X, Y in zip(X_list, Y_list):
                        Y_list_.append(self._get_residuals_matrix(X, Y, P, child))
                    Y_list = Y_list_

            if changed:
                t = 2
            else:
                t += 1
                if t > maxnum_vals:
                    break

        for i in range(d):
            non_parents = set()
            for j in P[i]:

                in_X_list = []
                in_Y_list = []
                for X in X_list:
                    n = X.shape[0]
                    residual_i = self._get_residual(X, i, P[i] - {j})
                    residual_j = self._get_residual(X, j, P[j])
                    in_X_list.append(np.reshape(residual_i, [n, 1]))
                    in_Y_list.append(np.reshape(residual_j, [n, 1]))

                if self._is_independent(in_X_list, in_Y_list):
                    non_parents.add(j)
            P[i] = P[i] - non_parents

        return P

    def _check_prior_knowledge(self, xj_list, xi):
        if self._pk_dict is not None:
            for xj in xj_list:
                if (xi in self._pk_dict) and (xj in self._pk_dict[xi]):
                    return True
        return False

    def _get_residuals_matrix(self, X, Y_old, P, child):
        Y = copy.deepcopy(Y_old)
        Y[:, child] = self._get_residual(X, child, P[child])
        return Y

    def _get_child(self, X_list, variables_set, P, N, Y_list):
        prev_independence = 0.0 if self._independence == "hsic" else 1.0
        max_independence_child = None

        for child in variables_set:
            parents = variables_set - {child}

            if self._check_prior_knowledge(parents, child):
                continue

            if not self._check_correlation(child, parents, N):
                continue

            in_X_list = []
            in_Y_list = []
            for X, Y in zip(X_list, Y_list):
                n = X.shape[0]
                residual = self._get_residual(X, child, parents | P[child])
                in_X_list.append(np.reshape(residual, [n, 1]))
                in_Y_list.append(np.reshape(Y[:, list(parents)], [n, len(parents)]))

            is_ind, value = self._is_independent_by(
                in_X_list, in_Y_list, prev_independence
            )
            if is_ind:
                prev_independence = value
                max_independence_child = child

        if self._independence == "hsic":
            is_independent = prev_independence > self._alpha
        elif self._independence == "fcorr":
            is_independent = prev_independence < self._ind_corr

        return max_independence_child, is_independent

    def _check_independence_withou_K(self, parents, child, P, N, Y_list):
        for parent in parents:

            in_X_list = []
            in_Y_list = []
            for Y in Y_list:
                n = Y.shape[0]
                in_X_list.append(np.reshape(Y[:, child], [n, 1]))
                in_Y_list.append(np.reshape(Y[:, parent], [n, 1]))

            if self._is_independent(in_X_list, in_Y_list):
                return False
        return True

    def _check_identified_causality(self, variables_set, P):
        variables_list = list(variables_set)
        for i in variables_list:
            for j in variables_list[variables_list.index(i) + 1 :]:
                if (j in P[i]) or (i in P[j]):
                    return False
        return True

    def _check_correlation(self, child, parents, N):
        for parent in parents:
            if parent not in N[child]:
                return False
        return True

    def _estimate_adjacency_matrix(self, X, P, U):
        B = np.zeros([X.shape[1], X.shape[1]], dtype="float64")
        for i, parents in enumerate(P):
            for parent in parents:
                B[i, parent] = 1
        for confounded_pair in U:
            confounded_pair = list(confounded_pair)
            B[confounded_pair[0], confounded_pair[1]] = np.nan
            B[confounded_pair[1], confounded_pair[0]] = np.nan
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
