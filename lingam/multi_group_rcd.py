"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""

import itertools
import numbers
import warnings

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import pearsonr, shapiro, chi2
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array, resample

from .bootstrap import BootstrapResult
from .hsic import get_gram_matrix, get_kernel_width, hsic_test_gamma, hsic_teststat
from .utils import f_correlation, calculate_total_effect


class MultiGroupRCD:
    """Implementation of RCD Algorithm with multiple groups"""

    def __init__(
        self,
        max_explanatory_num=2,
        cor_alpha=0.01,
        ind_alpha=0.01,
        shapiro_alpha=0.01,
        MLHSICR=True,
        bw_method="mdbs",
        independence="hsic",
        ind_corr=0.5,
    ):
        """Construct a model.

        Parameters
        ----------
        max_explanatory_num : int, optional (default=2)
            Maximum number of explanatory variables.
        cor_alpha : float, optional (default=0.01)
            Alpha level for pearson correlation.
        ind_alpha : float, optional (default=0.01)
            Alpha level for HSIC.
        shapiro_alpha : float, optional (default=0.01)
            Alpha level for Shapiro-Wilk test.
        MLHSICR : bool, optional (default=True)
            If True, use MLHSICR for multiple regression, if False, use OLS for multiple regression.
        bw_method : str, optional (default=``mdbs``)
                The method used to calculate the bandwidth of the HSIC.

            * ``mdbs`` : Median distance between samples.
            * ``scott`` : Scott's Rule of Thumb.
            * ``silverman`` : Silverman's Rule of Thumb.
        independence : {'hsic', 'fcorr'}, optional (default='hsic')
            Methods to determine independence. If 'hsic' is set, test for independence by HSIC.
            If 'fcorr' is set, independence is determined by F-correlation.
        ind_corr : float, optional (default=0.5)
            The threshold value for determining independence by F-correlation;
            independence is determined when the value of F-correlation is below this threshold value.
        """
        # Check parameters
        if max_explanatory_num <= 0:
            raise ValueError("max_explanatory_num must be > 0.")

        if cor_alpha < 0:
            raise ValueError("cor_alpha must be >= 0.")

        if ind_alpha < 0:
            raise ValueError("ind_alpha must be >= 0.")

        if shapiro_alpha < 0:
            raise ValueError("shapiro_alpha must be >= 0.")

        if bw_method not in ("mdbs", "scott", "silverman"):
            raise ValueError("bw_method must be 'mdbs', 'scott' or 'silverman'.")

        if independence not in ("hsic", "fcorr"):
            raise ValueError("independence must be 'hsic' or 'fcorr'.")

        if ind_corr < 0.0:
            raise ValueError("ind_corr must be an float greater than 0.")

        self._max_explanatory_num = max_explanatory_num
        self._cor_alpha = cor_alpha
        self._ind_alpha = ind_alpha
        self._shapiro_alpha = shapiro_alpha
        self._MLHSICR = MLHSICR
        self._bw_method = bw_method
        self._ancestors_list = None
        self._adjacency_matrices = None
        self._independence = independence
        self._ind_corr = ind_corr

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
        # Check parameters
        X_list = self._check_X_list(X_list)

        # Extract a set of ancestors of each variable
        M = self._extract_ancestors(X_list)

        # Extract parents (direct causes) from the set of ancestors.
        P = self._extract_parents(X_list, M)

        # Find the pairs of variables affected by the same latent confounders.
        C = self._extract_vars_sharing_confounders(X_list, P)

        self._ancestors_list = M

        self._adjacency_matrices = []
        for X in X_list:
            adjacency_matrix = self._estimate_adjacency_matrix(X, P, C)
            self._adjacency_matrices.append(adjacency_matrix)

        return self

    def estimate_total_effect(self, X_list, from_index, to_index):
        """Estimate total effect using causal model.

        Parameters
        ----------
        X_list : array-like, shape (X, ...)
            Multiple datasets for training, where ``X`` is an dataset.
            The shape of ''X'' is (n_samples, n_features),
            where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
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
        X_list = self._check_X_list(X_list)

        # Check from/to ancestors
        if to_index in self._ancestors_list[from_index]:
            warnings.warn(
                f"The estimated causal effect may be incorrect because "
                f"the causal order of the destination variable (to_index={to_index}) "
                f"is earlier than the source variable (from_index={from_index})."
            )

        effects = []
        for X, am in zip(X_list, self._adjacency_matrices):
            # Check confounders
            if True in np.isnan(am[from_index]):
                warnings.warn(
                    f"The estimated causal effect may be incorrect because "
                    f"the source variable (from_index={from_index}) is influenced by confounders."
                )
                return effects

            # from_index + parents indices
            parents = np.where(np.abs(am[from_index]) > 0)[0]
            predictors = [from_index]
            predictors.extend(parents)

            # Estimate total effect
            lr = LinearRegression()
            lr.fit(X[:, predictors], X[:, to_index])

            effects.append(lr.coef_[0])

        return effects

    def estimate_total_effect2(self, from_index, to_index):
        """Estimate total effect using causal model.

        Parameters
        ----------
        from_index :
            Index of source variable to estimate total effect.
        to_index :
            Index of destination variable to estimate total effect.

        Returns
        -------
        total_effect : float
            Estimated total effect.
        """
        # Check from/to ancestors
        if to_index in self._ancestors_list[from_index]:
            warnings.warn(
                f"The estimated causal effect may be incorrect because "
                f"the causal order of the destination variable (to_index={to_index}) "
                f"is earlier than the source variable (from_index={from_index})."
            )

        effects = []
        for am in self._adjacency_matrices:
            # Check confounders
            if True in np.isnan(am[from_index]):
                warnings.warn(
                    f"The estimated causal effect may be incorrect because "
                    f"the source variable (from_index={from_index}) is influenced by confounders."
                )
                return effects

            effect = calculate_total_effect(am, from_index, to_index)
            effects.append(effect)

        return effects

    def get_error_independence_p_values(self, X_list):
        """Calculate the p-value matrix of independence between error variables.

        Parameters
        ----------
        X_list : array-like, shape (X, ...)
            Multiple datasets for training, where ``X`` is an dataset.
            The shape of ''X'' is (n_samples, n_features),
            where ``n_samples`` is the number of samples and ``n_features`` is the number of features.

        Returns
        -------
        independence_p_values : array-like, shape (n_datasets, n_features, n_features)
            p-value matrix of independence between error variables.
        """
        # Check parameters
        X_list = self._check_X_list(X_list)

        p_values = np.zeros([len(X_list), self._n_features, self._n_features])
        for d, (X, am) in enumerate(zip(X_list, self._adjacency_matrices)):
            n_samples = X.shape[0]
            E = X - np.dot(am, X.T).T
            nan_cols = list(set(np.argwhere(np.isnan(am)).ravel()))
            for i, j in itertools.combinations(range(self._n_features), 2):
                if i in nan_cols or j in nan_cols:
                    p_values[d, i, j] = np.nan
                    p_values[d, j, i] = np.nan
                else:
                    _, p_value = hsic_test_gamma(
                        np.reshape(E[:, i], [n_samples, 1]),
                        np.reshape(E[:, j], [n_samples, 1]),
                    )
                    p_values[d, i, j] = p_value
                    p_values[d, j, i] = p_value

        return p_values

    def bootstrap(self, X_list, n_sampling):
        """Evaluate the statistical reliability of DAG based on the bootstrapping.

        Parameters
        ----------
        X_list : array-like, shape (X, ...)
            Multiple datasets for training, where ``X`` is an dataset.
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
        X_list = self._check_X_list(X_list)

        if isinstance(n_sampling, (numbers.Integral, np.integer)):
            if not 0 < n_sampling:
                raise ValueError("n_sampling must be an integer greater than 0.")
        else:
            raise ValueError("n_sampling must be an integer greater than 0.")

        # Bootstrapping
        adjacency_matrices_list = np.zeros(
            [len(X_list), n_sampling, self._n_features, self._n_features]
        )
        total_effects_list = np.zeros(
            [len(X_list), n_sampling, self._n_features, self._n_features]
        )
        for n in range(n_sampling):
            resampled_X_list = [resample(X) for X in X_list]
            self.fit(resampled_X_list)

            for i, am in enumerate(self._adjacency_matrices):
                adjacency_matrices_list[i][n] = am

            # Calculate total effects
            for to, ancestors in enumerate(self._ancestors_list):
                for from_ in ancestors:
                    effects = self.estimate_total_effect2(from_, to)
                    for i, effect in enumerate(effects):
                        total_effects_list[i, n, to, from_] = effect

        result_list = []
        for am, te in zip(adjacency_matrices_list, total_effects_list):
            result_list.append(BootstrapResult(am, te))

        return result_list

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

    def _get_common_ancestors(self, M, U):
        """Get the set of common ancestors of U"""
        Mj_list = [M[xj] for xj in U]
        return set.intersection(*Mj_list)

    def _get_resid_and_coef(self, X, endog_idx, exog_idcs):
        """Get the residuals and coefficients of the ordinary least squares method"""
        lr = LinearRegression()
        lr.fit(X[:, exog_idcs], X[:, endog_idx])
        resid = X[:, endog_idx] - lr.predict(X[:, exog_idcs])
        return resid, lr.coef_

    def _get_residual_matrix(self, X, U, H_U):
        if len(H_U) == 0:
            return X

        Y = np.zeros_like(X)
        for xj in U:
            Y[:, xj], _ = self._get_resid_and_coef(X, xj, list(H_U))
        return Y

    def _is_non_gaussianity(self, Y_list, U):
        """Test whether a variable is generated from a non-Gaussian process using the Shapiro-Wilk test"""
        for xj in U:
            fisher_stat = 0
            for Y in Y_list:
                shapiro_p = shapiro(Y[:, xj])[1]
                fisher_stat += np.inf if shapiro_p == 0 else -2 * np.log(shapiro_p)

            fisher_p = chi2.sf(fisher_stat, df=2 * self._k)

            if fisher_p > self._shapiro_alpha:
                return False

        return True

    def _is_correlated(self, Y_list, U):
        # Estimate that the two variables are linearly correlated using the Pearson's correlation
        for xi, xj in itertools.combinations(U, 2):
            fisher_stat = 0
            for Y in Y_list:
                corr_p = pearsonr(Y[:, xi], Y[:, xj])[1]
                fisher_stat += np.inf if corr_p == 0 else -2 * np.log(corr_p)

            fisher_p = chi2.sf(fisher_stat, df=2 * self._k)

            if fisher_p >= self._cor_alpha:
                return False
        return True

    def _exists_ancestor_in_U(self, M, U, xi, xj_list):
        # Check xi is not in Mj, the ancestor of xj.
        for xj in xj_list:
            if xi in M[xj]:
                return True

        # Check if xj_list is a subset of Mi, the ancestor of xi.
        if set(xj_list) == set(xj_list) & M[xi]:
            return True
        return False

    def _is_independent(self, X_list, Y_list, xj):
        if self._independence == "hsic":
            fisher_stat = 0
            for i, Y in enumerate(Y_list):
                _, hsic_p = hsic_test_gamma(
                    X_list[i], Y[:, xj], bw_method=self._bw_method
                )
                fisher_stat += np.inf if hsic_p == 0 else -2 * np.log(hsic_p)

            fisher_p = chi2.sf(fisher_stat, df=2 * self._k)

            is_independent = fisher_p > self._ind_alpha

        elif self._independence == "fcorr":
            max_f_corr = 0.0
            for i, Y in enumerate(Y_list):
                f_corr = f_correlation(X_list[i], Y[:, xj])
                if f_corr > max_f_corr:
                    max_f_corr = f_corr

            is_independent = f_corr < self._ind_corr

        return is_independent

    def _is_independent_of_resid(self, Y_list, xi, xj_list):
        """Check whether the residuals obtained from multiple regressions are independent"""
        # Multiple Regression with OLS.
        resid_list = []
        for Y in Y_list:
            resid, _ = self._get_resid_and_coef(Y, xi, xj_list)
            resid_list.append(resid)

        is_all_independent = True
        for xj in xj_list:
            is_independent = self._is_independent(resid_list, Y_list, xj)
            if not is_independent:
                is_all_independent = False
                break

        if is_all_independent:
            return True
        elif len(xj_list) == 1 or self._MLHSICR is False:
            return False

        # Multiple Regression with MLHSICR.
        resid_list = []
        for Y in Y_list:
            resid, _ = self._get_resid_and_coef_by_MLHSICR(Y, xi, xj_list)
            resid_list.append(resid)

        for xj in xj_list:
            is_independent = self._is_independent(resid_list, Y_list, xj)
            if not is_independent:
                return False

        return True

    def _get_resid_and_coef_by_MLHSICR(self, Y, xi, xj_list):
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

        _, initial_coef = self._get_resid_and_coef(Y, xi, xj_list)
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

    def _extract_ancestors(self, X_list):
        """Extract a set of ancestors of each variable"""
        M = [set() for i in range(self._n_features)]
        l = 1
        hu_history = {}

        while True:
            changed = False
            U_list = itertools.combinations(range(self._n_features), l + 1)
            for U in U_list:
                U = list(U)
                U.sort()

                # Get the set of common ancestors of U
                H_U = self._get_common_ancestors(M, U)

                if tuple(U) in hu_history and H_U == hu_history[tuple(U)]:
                    continue

                # Y_list = np.zeros(X_list.shape)
                Y_list = []
                for X in X_list:
                    Y_list.append(self._get_residual_matrix(X, U, H_U))

                # Test whether a variable is generated from a non-Gaussian process using the Shapiro-Wilk test
                if not self._is_non_gaussianity(Y_list, U):
                    continue

                # Estimate that the two variables are linearly correlated using the Pearson's correlation
                if not self._is_correlated(Y_list, U):
                    continue

                sink_set = []
                for xi in U:
                    xj_list = list(set(U) - set([xi]))
                    if self._exists_ancestor_in_U(M, U, xi, xj_list):
                        continue

                    # Check whether the residuals obtained from multiple regressions are independent
                    if self._is_independent_of_resid(Y_list, xi, xj_list):
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
            elif l < self._max_explanatory_num:
                l += 1
            else:
                break

        return M

    def _extract_parents(self, X_list, M):
        """Extract parents (direct causes) from a set of ancestors"""
        P = [set() for i in range(self._n_features)]

        for xi in range(self._n_features):
            for xj in M[xi]:
                # Check if xj is the parent of xi
                if self._is_parent(X_list, M, xj, xi):
                    P[xi].add(xj)

        return P

    def _is_parent(self, X_list, M, xj, xi):
        # Check if zi and wj are correlated
        fisher_stat = 0
        for X in X_list:
            if len(M[xi] - set([xj])) > 0:
                zi, _ = self._get_resid_and_coef(X, xi, list(M[xi] - set([xj])))
            else:
                zi = X[:, xi]

            if len(M[xi] & M[xj]) > 0:
                wj, _ = self._get_resid_and_coef(X, xj, list(M[xi] & M[xj]))
            else:
                wj = X[:, xj]

            corr_p = pearsonr(wj, zi)[1]
            fisher_stat += np.inf if corr_p == 0 else -2 * np.log(corr_p)

        fisher_p = chi2.sf(fisher_stat, df=2 * self._k)

        return fisher_p < self._cor_alpha

    def _get_resid_to_parent(self, X, idx, P):
        if len(P[idx]) == 0:
            return X[:, idx]

        resid, _ = self._get_resid_and_coef(X, idx, list(P[idx]))
        return resid

    def _extract_vars_sharing_confounders(self, X_list, P):
        """Find the pairs of variables affected by the same latent confounders."""
        C = [set() for i in range(self._n_features)]

        for i, j in itertools.combinations(range(self._n_features), 2):
            if (i in P[j]) or (j in P[i]):
                continue

            fisher_stat = 0
            for X in X_list:
                resid_xi = self._get_resid_to_parent(X, i, P)
                resid_xj = self._get_resid_to_parent(X, j, P)
                corr_p = pearsonr(resid_xi, resid_xj)[1]
                fisher_stat += np.inf if corr_p == 0 else -2 * np.log(corr_p)

            fisher_p = chi2.sf(fisher_stat, df=2 * self._k)

            if fisher_p < self._cor_alpha:
                C[i].add(j)
                C[j].add(i)

        return C

    def _estimate_adjacency_matrix(self, X, P, C):
        # Check parents
        B = np.zeros([self._n_features, self._n_features], dtype="float64")
        for xi in range(self._n_features):
            xj_list = list(P[xi])
            xj_list.sort()
            if len(xj_list) == 0:
                continue

            _, coef = self._get_resid_and_coef(X, xi, xj_list)
            for j, xj in enumerate(xj_list):
                B[xi, xj] = coef[j]

        # Check confounders
        for xi in range(self._n_features):
            xj_list = list(C[xi])
            xj_list.sort()
            if len(xj_list) == 0:
                continue

            for xj in xj_list:
                B[xi, xj] = np.nan

        return B

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

    @property
    def ancestors_list_(self):
        """Estimated ancestors list.

        Returns
        -------
        ancestors_list_ : array-like, shape (n_features)
            The list of causal ancestors sets, where
            n_features is the number of features.
        """
        return self._ancestors_list
