"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""

import numbers

import numpy as np
from scipy.stats import laplace
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array, resample

from .utils import predict_adaptive_lasso, calculate_total_effect
from .bootstrap import BootstrapResult


class GroupDirectLiNGAM:
    """Implementation of GroupDirectLiNGAM Algorithm [1]_

    References
    ----------
    .. [1] D. Entner and P. O. Hoyer. Estimating a causal order among groups of variables in linear models.
       In Proc. 22nd International Conference on Artificial Neural Networks (ICANN2012),
       pp. 83--90, Lausanne, Switzerland, 2012.
    """
    def __init__(self, prior_knowledge=None):
        """Construct a GroupDirectLiNGAM model.

        Parameters
        ----------
        prior_knowledge : array-like, shape (n_groups, n_groups), optional (default=None)
            Prior knowledge used for causal discovery, where ``n_groups`` is the number of groups.

            The elements of prior knowledge matrix are defined as follows [1]_:

            * ``0`` : :math:`x_i` does not have a directed path to :math:`x_j`
            * ``1`` : :math:`x_i` has a directed path to :math:`x_j`
            * ``-1`` : No prior knowledge is available to know if either of the two cases above (0 or 1) is true.
        """
        self._adjacency_matrix = None
        self._causal_order = []

        self._Aknw = prior_knowledge
        if self._Aknw is not None:
            self._Aknw = check_array(self._Aknw)
            self._Aknw = np.where(self._Aknw < 0, np.nan, self._Aknw)

    def fit(self, X, groups):
        """Fit the model to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.
        groups : array-like, shape (n_groups)
            The list of features for each group.
            where ``n_groups`` is the number of groups.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Check parameters
        X = check_array(X)
        n_groups = len(groups)

        # Check prior knowledge
        if self._Aknw is not None:
            if (n_groups, n_groups) != self._Aknw.shape:
                raise ValueError(
                    "The shape of prior knowledge must be (n_groups, n_groups)"
                )
            else:
                # Extract all partial orders in prior knowledge matrix
                self._partial_orders = self._extract_partial_orders(self._Aknw)

        # Causal discovery
        self._causal_order = self._estimate_causal_order(X, groups)

        return self._estimate_adjacency_matrix(X, groups, prior_knowledge=self._Aknw)

    def _extract_partial_orders(self, pk):
        """Extract partial orders from prior knowledge."""
        path_pairs = np.array(np.where(pk == 1)).transpose()
        no_path_pairs = np.array(np.where(pk == 0)).transpose()

        # Check for inconsistencies in pairs with path
        check_pairs = np.concatenate([path_pairs, path_pairs[:, [1, 0]]])
        if len(check_pairs) > 0:
            pairs, counts = np.unique(check_pairs, axis=0, return_counts=True)
            if len(pairs[counts > 1]) > 0:
                raise ValueError(
                    f"The prior knowledge contains inconsistencies at the following indices: {pairs[counts>1].tolist()}"
                )

        # Check for inconsistencies in pairs without path
        # If there are duplicate pairs without path, they cancel out and are not ordered.
        check_pairs = np.concatenate([no_path_pairs, no_path_pairs[:, [1, 0]]])
        if len(check_pairs) > 0:
            pairs, counts = np.unique(check_pairs, axis=0, return_counts=True)
            check_pairs = np.concatenate([no_path_pairs, pairs[counts > 1]])
            pairs, counts = np.unique(check_pairs, axis=0, return_counts=True)
            no_path_pairs = pairs[counts < 2]

        check_pairs = np.concatenate([path_pairs, no_path_pairs[:, [1, 0]]])
        if len(check_pairs) == 0:
            # If no pairs are extracted from the specified prior knowledge,
            return check_pairs

        pairs = np.unique(check_pairs, axis=0)
        return pairs[:, [1, 0]]  # [to, from] -> [from, to]

    def _search_candidate(self, S):
        """Search for candidate features"""
        # If no prior knowledge is specified, nothing to do.
        if self._Aknw is None:
            return S

        # Candidate features that are not to the left of the partial orders
        if len(self._partial_orders) != 0:
            Sc = [i for i in S if i not in self._partial_orders[:, 1]]
            return Sc

        return S

    def _get_resid_and_coef(self, predictors, target):
        """Get the residuals and coefficients of the ordinary least squares method"""
        lr = LinearRegression().fit(predictors, target)
        resid = target - lr.predict(predictors)
        return resid, lr.coef_

    def _log_likelihood(self, x, y):
        """Compute the log likelihoods assuming a Laplace distribution"""
        residuals, _ = self._get_resid_and_coef(y, x)
        sigma = np.mean(np.abs(residuals))
        return np.sum(laplace.logpdf(residuals, loc=0, scale=sigma))

    def _log_likelihood_ratio(self, x, y):
        """Compute the (normalized) ratio of the log likelihoods for the two causal models"""
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        m = len(x)
        LogL_xy = self._log_likelihood(x, y)
        LogL_yx = self._log_likelihood(y, x)

        R = (LogL_xy - LogL_yx) / m
        return R

    def _pairwise_measure(self, X, remaining_groups, j, groups):
        """Compute the pairwise measure between one group and all remaining groups"""
        mu = 0
        total_n_i = 0
        X_j = X[:, groups[j]]
        n_j = X_j.shape[1]
        for i in remaining_groups:
            if i == j:
                continue
            X_i = X[:, groups[i]]
            n_i = X_i.shape[1]
            total_n_i += n_i
            for k in range(n_j):
                x_jk = X_j[:, k]
                for l in range(n_i):
                    x_il = X_i[:, l]
                    r_li, coef = self._get_resid_and_coef(X_j, x_il)
                    b_lk = coef[k]
                    z_ikl = b_lk * x_jk + r_li
                    R = self._log_likelihood_ratio(x_jk, z_ikl)
                    mu += min(0, R) ** 2
        return mu / (n_j * total_n_i)

    def _get_residual_matrix(self, X, predictors, targets):
        """Get the matrix of residuals"""
        if len(predictors) == 0:
            return X

        Y = np.zeros_like(X)
        for target in targets:
            Y[:, target], _ = self._get_resid_and_coef(X[:, list(predictors)], X[:, target])
        return Y

    def _estimate_causal_order(self, X, groups):
        """Estimate the causal order of groups by repeatedly identifying exogenous groups"""
        n_groups = len(groups)
        G = np.arange(n_groups)
        Rj = np.copy(X)
        S = G
        K = []
        for _ in range(n_groups):
            Sc = self._search_candidate(S)

            if len(Sc) == 1:
                k = Sc[0]
            else:
                # Find an exogenous group
                mu_list = [float('inf') for j in range(n_groups)]
                for j in G:
                    if j in Sc:
                        mu = self._pairwise_measure(Rj, Sc, j, groups)
                        mu_list[j] = mu

                k = np.argmin(mu_list)

            S = S[S != k]
            K.append(k)

            # Replace the data matrix with the residuals
            Xj = groups[k]
            Xi = [k for i in S for k in groups[i]]
            Rj = self._get_residual_matrix(Rj, Xj, Xi)

            # Update partial orders
            if self._Aknw is not None:
                self._partial_orders = self._partial_orders[self._partial_orders[:, 0] != k]

        return K

    def _estimate_adjacency_matrix(self, X, groups, prior_knowledge=None):
        """Estimate adjacency matrix by causal order.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        groups : array-like, shape (n_groups)
            The list of features for each group.
        prior_knowledge : array-like, shape (n_groups, n_groups), optional (default=None)
            Prior knowledge matrix.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if prior_knowledge is not None:
            pk = prior_knowledge.copy()
            np.fill_diagonal(pk, 0)

        B = np.zeros([X.shape[1], X.shape[1]], dtype="float64")
        for i in range(1, len(self._causal_order)):
            target_group_idx = self._causal_order[i]
            predictor_groups_idxs = self._causal_order[:i]

            # Exclude groups specified in no_path with prior knowledge
            if prior_knowledge is not None:
                predictor_groups_idxs = [p for p in predictor_groups_idxs if pk[target_group_idx, p] != 0]

            # target is exogenous groups if predictors are empty
            if len(predictor_groups_idxs) == 0:
                continue

            # Retrieve variables from groups
            target_group = groups[target_group_idx]
            predictor_groups = [groups[idx] for idx in predictor_groups_idxs]
            predictors = [var for group in predictor_groups for var in group]

            for target in target_group:
                B[target, predictors] = predict_adaptive_lasso(X, predictors, target)

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

    @property
    def causal_order_(self):
        """Estimated causal ordering.

        Returns
        -------
        causal_order_ : array-like, shape (n_groups)
            The causal order of fitted model, where
            n_groups is the number of groups.
        """
        return self._causal_order

    def bootstrap(self, X, groups, n_sampling):
        """Evaluate the statistical reliability of DAG based on the bootstrapping.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.
        groups : array-like, shape (n_groups)
            The list of features for each group.
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
                raise ValueError("n_sampling must be an integer greater than 0.")
        else:
            raise ValueError("n_sampling must be an integer greater than 0.")

        # Bootstrapping
        adjacency_matrices = np.zeros([n_sampling, X.shape[1], X.shape[1]])
        total_effects = np.zeros([n_sampling, X.shape[1], X.shape[1]])
        index = np.arange(X.shape[0])
        resampled_indices = []
        for i in range(n_sampling):
            resampled_X, resampled_index = resample(X, index)
            self.fit(resampled_X, groups)
            adjacency_matrices[i] = self._adjacency_matrix

            # Calculate total effects
            for c, from_group in enumerate(self._causal_order):
                for to_group in self._causal_order[c + 1 :]:
                    for from_ in groups[from_group]:
                        for to in groups[to_group]:
                            total_effects[i, to, from_] = calculate_total_effect(
                                self._adjacency_matrix, from_, to
                            )

            resampled_indices.append(resampled_index)

        return BootstrapResult(adjacency_matrices, total_effects, resampled_indices=resampled_indices)
