"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""


from .hsic import hsic_test_gamma
import numpy as np
import itertools
from pygam import LinearGAM
import copy
from sklearn.utils import check_array


class CAMUV:
    """Implementation of CAM-UV Algorithm [1]_

    References
    ----------
    .. [1] T.N.Maeda and S.Shimizu. Causal additive models with unobserved variables.
       In Proc. Thirty-Seventh Conference on Uncertainty in Artificial Intelligence (UAI). PMLR  161:97-106, 2021.
    """

    def __init__(
            self,
            alpha=0.01,
            num_explanatory_vals=2
    ):

        """Construct a CAM-UV model.

        Parameters
        ----------
         alpha : float, optional (default=0.01)
             Alpha level.
         num_explanatory_vals : int, optional (default=2)
             Maximum number of explanatory variables.
        """

        # Check parameters
        if num_explanatory_vals <= 0:
            raise ValueError("num_explanatory_vals must be > 0.")

        if alpha < 0:
            raise ValueError("alpha must be >= 0.")

        self._num_explanatory_vals = num_explanatory_vals
        self._alpha = alpha

    def fit(self, X):
        X = check_array(X)

        n = X.shape[0]
        d = X.shape[1]
        N = self._get_neighborhoods(X, self._alpha)
        P = self._find_parents(X, self._alpha, self._num_explanatory_vals, N)

        U = []

        for i in range(d):
            for j in range(d)[i+1:]:
                if (i in P[j]) or (j in P[i]):
                    continue
                if (i not in N[j]) or (j not in N[i]):
                    continue
                i_residual = self._get_residual(X, i, P[i])
                j_residual = self._get_residual(X, j, P[j])
                in_X = np.reshape(i_residual, [n, 1])
                in_Y = np.reshape(j_residual, [n, 1])
                independence = hsic_test_gamma(X=in_X, Y=in_Y)[1]
                if independence < self._alpha:
                    if not set([i, j]) in U:
                        U.append(set([i, j]))

        return self._estimate_adjacency_matrix(X, P, U)

    def _get_residual(self, X, explained_i, explanatory_ids):

        explanatory_ids = list(explanatory_ids)

        if len(explanatory_ids) == 0:
            residual = X[:, explained_i]
        else:
            gam = LinearGAM().fit(X[:, explanatory_ids], X[:, explained_i])
            residual = X[:, explained_i] - gam.predict(X[:, explanatory_ids])
        return residual

    def _get_neighborhoods(self, X, alpha):
        n = X.shape[0]
        d = X.shape[1]
        N = [set() for i in range(d)]
        for i in range(d):
            for j in range(d)[i+1:]:
                in_X = np.reshape(X[:, i], [n, 1])
                in_Y = np.reshape(X[:, j], [n, 1])
                independence = hsic_test_gamma(X=in_X, Y=in_Y)[1]
                if independence < self._alpha:
                    N[i].add(j)
                    N[j].add(i)
        return N

    def _find_parents(self, X, alpha, maxnum_vals, N):
        n = X.shape[0]
        d = X.shape[1]
        P = [set() for i in range(d)]  # Parents
        t = 2
        Y = copy.deepcopy(X)

        while (True):
            changed = False
            variables_set_list = list(itertools.combinations(set(range(d)), t))
            for variables_set in variables_set_list:
                variables_set = set(variables_set)

                if not self._check_identified_causality(variables_set, P):
                    continue

                child, independence_with_K = self._get_child(X, variables_set, P, N, Y, alpha)
                if not independence_with_K > alpha:
                    continue

                parents = variables_set-{child}
                if not self._check_independence_withou_K(parents, child, P, N, Y, alpha):
                    continue

                for parent in parents:
                    P[child].add(parent)
                    changed = True
                    Y = self._get_residuals_matrix(X, Y, P, child)

            if changed:
                t = 2
            else:
                t += 1
                if t > maxnum_vals:
                    break

        for i in range(d):
            non_parents = set()
            for j in P[i]:
                residual_i = self._get_residual(X, i, P[i]-{j})
                residual_j = self._get_residual(X, j, P[j])
                in_X = np.reshape(residual_i, [n, 1])
                in_Y = np.reshape(residual_j, [n, 1])
                independence = hsic_test_gamma(X=in_X, Y=in_Y)[1]
                if independence > alpha:
                    non_parents.add(j)
            P[i] = P[i] - non_parents

        return P

    def _get_residuals_matrix(self, X, Y_old, P, child):

        Y = copy.deepcopy(Y_old)
        Y[:, child] = self._get_residual(X, child, P[child])
        return Y

    def _get_child(self, X, variables_set, P, N, Y, alpha):
        n = X.shape[0]

        max_independence = 0.0
        max_independence_child = None

        for child in variables_set:
            parents = variables_set - {child}

            if not self._check_correlation(child, parents, N):
                continue

            residual = self._get_residual(X, child, parents | P[child])
            in_X = np.reshape(residual, [n, 1])
            in_Y = np.reshape(Y[:, list(parents)], [n, len(parents)])
            independence = hsic_test_gamma(X=in_X, Y=in_Y)[1]
            if max_independence < independence:
                max_independence = independence
                max_independence_child = child

        return max_independence_child, max_independence

    def _check_independence_withou_K(self, parents, child, P, N, Y, alpha):
        n = Y.shape[0]
        for parent in parents:
            in_X = np.reshape(Y[:, child], [n, 1])
            in_Y = np.reshape(Y[:, parent], [n, 1])
            independence = hsic_test_gamma(X=in_X, Y=in_Y)[1]
            if alpha < independence:
                return False
        return True

    def _check_identified_causality(self, variables_set, P):
        variables_list = list(variables_set)
        for i in variables_list:
            for j in variables_list[variables_list.index(i)+1:]:
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
