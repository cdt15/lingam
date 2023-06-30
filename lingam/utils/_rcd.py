import itertools

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import pearsonr, shapiro
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array

from ..hsic import get_gram_matrix, get_kernel_width, hsic_test_gamma, hsic_teststat


def extract_ancestors(
    X,
    max_explanatory_num=2,
    cor_alpha=0.01,
    ind_alpha=0.01,
    shapiro_alpha=0.01,
    MLHSICR=True,
    bw_method="mdbs",
):
    """Extract a set of ancestors of each variable
    Implementation of RCD Algorithm1 [1]_

    References
    ----------
    .. [1] T.N.Maeda and S.Shimizu. RCD: Repetitive causal discovery of
        linear non-Gaussian acyclic models with latent confounders.
        In Proc. 23rd International Conference on Artificial Intelligence and
        Statistics (AISTATS2020), Palermo, Sicily, Italy. PMLR  108:735-745, 2020.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where ``n_samples`` is the number of samples
        and ``n_features`` is the number of features.
    max_explanatory_num : int, optional (default=2)
        Maximum number of explanatory variables.
    cor_alpha : float, optional (default=0.01)
        Alpha level for pearson correlation.
    ind_alpha : float, optional (default=0.01)
        Alpha level for HSIC.
    shapiro_alpha : float, optional (default=0.01)
        Alpha level for Shapiro-Wilk test.
    MLHSICR : bool, optional (default=False)
        If True, use MLHSICR for multiple regression, if False, use OLS for multiple regression.
    bw_method : str, optional (default=``mdbs``)
            The method used to calculate the bandwidth of the HSIC.

        * ``mdbs`` : Median distance between samples.
        * ``scott`` : Scott's Rule of Thumb.
        * ``silverman`` : Silverman's Rule of Thumb.
    """
    n_features = X.shape[1]
    M = [set() for i in range(n_features)]
    l = 1
    hu_history = {}

    X = check_array(X)

    while True:
        changed = False
        U_list = itertools.combinations(range(n_features), l + 1)
        U_list = [sorted(list(U)) for U in U_list]

        for U in U_list:
            # Get the set of common ancestors of U
            H_U = get_common_ancestors(M, U)

            if tuple(U) in hu_history and H_U == hu_history[tuple(U)]:
                continue

            Y = get_residual_matrix(X, U, H_U)

            # Shapiro-Wilk test
            if not is_non_gaussianity(Y, U, shapiro_alpha):
                continue

            # Pearson's correlation
            is_cor = True
            for xi, xj in itertools.combinations(U, 2):
                if not is_correlated(Y[:, xi], Y[:, xj], cor_alpha):
                    is_cor = False
                    break
            if not is_cor:
                continue

            sink_set = []
            for xi in U:
                xj_list = list(set(U) - set([xi]))

                if exists_ancestor_in_U(M, U, xi, xj_list):
                    continue

                # Check whether the residuals obtained from multiple regressions are independent
                if is_independent_of_resid(
                    Y, xi, xj_list, ind_alpha, bw_method, MLHSICR
                ):
                    sink_set.append(xi)

            if len(sink_set) == 1:
                xi = sink_set[0]
                xj_list = list(set(U) - set(sink_set))

                if not M[xi] == M[xi] | set(xj_list):
                    M[xi] = M[xi] | set(xj_list)
                    changed = True

            hu_history[tuple(U)] = H_U

        if changed:
            l = 1
        elif l < max_explanatory_num:
            l += 1
        else:
            break

    return M


def get_common_ancestors(M, U):
    """Get the set of common ancestors of U"""
    Mj_list = [M[xj] for xj in U]
    return set.intersection(*Mj_list)


def get_residual_matrix(X, U, H_U):
    if len(H_U) == 0:
        return X

    Y = np.zeros_like(X)
    for xj in U:
        Y[:, xj], _ = get_resid_and_coef(X, xj, list(H_U))
    return Y


def is_non_gaussianity(Y, U, alpha):
    """Test whether a variable is generated from a non-Gaussian process using the Shapiro-Wilk test"""
    for xj in U:
        if shapiro(Y[:, xj])[1] > alpha:
            return False
    return True


def is_correlated(a, b, alpha):
    """Estimate that the two variables are linearly correlated using the Pearson's correlation"""
    return pearsonr(a, b)[1] < alpha


def exists_ancestor_in_U(M, U, xi, xj_list):
    # Check xi is not in Mj, the ancestor of xj.
    for xj in xj_list:
        if xi in M[xj]:
            return True

    # Check if xj_list is a subset of Mi, the ancestor of xi.
    if set(xj_list) == set(xj_list) & M[xi]:
        return True
    return False


def is_independent_of_resid(Y, xi, xj_list, alpha, bw_method, MLHSICR):
    """Check whether the residuals obtained from multiple regressions are independent"""
    n_samples = Y.shape[0]

    # Multiple Regression with OLS.
    is_all_independent = True
    resid, _ = get_resid_and_coef(Y, xi, xj_list)
    for xj in xj_list:
        if not is_independent(
            np.reshape(resid, [n_samples, 1]),
            np.reshape(Y[:, xj], [n_samples, 1]),
            bw_method,
            alpha,
        ):
            is_all_independent = False
            break

    if is_all_independent:
        return True
    elif len(xj_list) == 1 or MLHSICR is False:
        return False

    # Multiple Regression with MLHSICR.
    resid, _ = get_resid_and_coef_by_MLHSICR(Y, xi, xj_list)
    for xj in xj_list:
        if not is_independent(
            np.reshape(resid, [n_samples, 1]),
            np.reshape(Y[:, xj], [n_samples, 1]),
            bw_method,
            alpha,
        ):
            return False
    return True


def get_resid_and_coef(X, endog_idx, exog_idcs):
    """Get the residuals and coefficients of the ordinary least squares method"""
    lr = LinearRegression()
    lr.fit(X[:, exog_idcs], X[:, endog_idx])
    resid = X[:, endog_idx] - lr.predict(X[:, exog_idcs])
    return resid, lr.coef_


def get_resid_and_coef_by_MLHSICR(Y, xi, xj_list):
    """Get the residuals and coefficients by minimizing the sum of HSICs using the L-BFGS method."""
    n_samples = Y.shape[0]
    width_list = []
    Lc_list = []
    for xj in xj_list:
        yj = np.reshape(Y[:, xj], [n_samples, 1])
        width_xj = get_kernel_width(yj)
        _, Lc = get_gram_matrix(yj, width_xj)

        width_list.append(width_xj)
        Lc_list.append(Lc)

    _, initial_coef = get_resid_and_coef(Y, xi, xj_list)
    width_xi = get_kernel_width(np.reshape(Y[:, xi], [n_samples, 1]))

    # Calculate the sum of the Hilbert-Schmidt independence criterion
    def sum_empirical_hsic(coef):
        resid = Y[:, xi]
        width = width_xi
        for j, xj in enumerate(xj_list):
            resid = resid - coef[j] * Y[:, xj]
            width = width - coef[j] * width_list[j]
        _, Kc = get_gram_matrix(np.reshape(resid, [n_samples, 1]), width)

        objective = 0.0
        for j, xj in enumerate(xj_list):
            objective += hsic_teststat(Kc, Lc_list[j], n_samples)
        return objective

    # Estimate coefficients by minimizing the sum of HSICs using the L-BFGS method.
    coefs, _, _ = fmin_l_bfgs_b(
        func=sum_empirical_hsic, x0=initial_coef, approx_grad=True
    )

    resid = Y[:, xi]
    for j, xj in enumerate(xj_list):
        resid = resid - coefs[j] * Y[:, xj]
    return resid, coefs


def is_independent(X, Y, bw_method, alpha):
    _, p = hsic_test_gamma(X, Y, bw_method=bw_method)
    return p > alpha
