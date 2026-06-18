"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""

import warnings
import numbers

import numpy as np
from scipy.stats import chi2
from sklearn.utils import check_array

from .hsic import hsic_test_gamma
from .utils import find_all_paths


class LongitudinalRESIT:
    """Implementation of RESIT(regression with subsequent independence test) Algorithm [1]_ for longitudinal data

    References
    ----------
    .. [1] Jonas Peters, Joris M Mooij, Dominik Janzing, and Bernhard Sch ̈olkopf.
       Causal discovery with continuous additive noise models.
       Journal of Machine Learning Research, 15:2009-2053, 2014.

    Notes
    -----
    RESIT algorithm returns an **adjacency matrix consisting of zeros or ones**,
    rather than an adjacency matrix consisting of causal coefficients,
    in order to estimate nonlinear causality.
    """

    def __init__(
        self,
        regressor,
        n_lags=1,
        prior_knowledge=None,
        prior_knowledge_lag=None,
        alpha=0.01,
        is_common_graph=True,
    ):
        """Construct a LongitudinalRESIT model.

        Parameters
        ----------
        regressor : regressor object implementing 'fit' and 'predict' function (default=None)
            Regressor to compute residuals.
            This regressor object must have ``fit`` method and ``predict`` function like scikit-learn's model.
        n_lags : int, optional (default=1)
            Number of lags (past time points) to use. Must be an integer >= 1.
        prior_knowledge : array-like, optional (default=None)
            Prior knowledge used for the instantaneous causal graph,
            assuming a common causal structure across all time points.
            A single matrix of shape (n_vars, n_vars).
            Values: 0=no edge, 1=edge exists, -1=unknown.
        prior_knowledge_lag : array-like, optional (default=None)
            Prior knowledge used for the lag causal graph,
            assuming a common causal structure across all time points.
            A single tensor of shape (n_lags, n_vars, n_vars).
            Values: 0=no edge, 1=edge exists, -1=unknown.
        alpha : float, optional (default=0.01)
            Alpha level for HSIC independence test when removing superfluous edges.
        is_common_graph : bool, optional (default=True)
            If True, estimate a single causal graph shared across all time points.
            If False, estimate an independent causal graph for each time point.
        """
        if regressor is None:
            raise ValueError("Specify regression model in 'regressor'.")
        if not (hasattr(regressor, "fit") and hasattr(regressor, "predict")):
            raise ValueError("'regressor' has no fit or predict method.")
        if not isinstance(alpha, numbers.Number) or alpha < 0.0:
            raise ValueError("alpha must be a float greater than 0.")
        if not isinstance(n_lags, int) or n_lags < 1:
            raise ValueError("n_lags must be an integer greater than or equal to 1.")

        self._reg = regressor
        self._n_lags = n_lags
        self._alpha = alpha
        self._is_common_graph = is_common_graph
        self._causal_order = None
        self._B_list = None
        self._A_list = None
        self._adjacency_matrices = None
        self._prior_knowledge_raw = prior_knowledge
        self._prior_knowledge_lag_raw = prior_knowledge_lag
        self._Aknw = None
        self._Aknw_lag = None
        self._partial_orders = None

    def fit(self, X_list):
        """Fit the model to X_list.

        Parameters
        ----------
        X_list : list of ndarray, shape [(n_samples, n_vars), ...]
            Multiple datasets for training, where each dataset corresponds to a time point.
            The length of X_list is T (number of time points). T >= 2 is required.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X_list = self._validate_X_list(X_list)
        n_vars = X_list[0].shape[1]
        T = len(X_list)

        self._validate_prior_knowledge(n_vars, T)

        causal_order = self._estimate_causal_order(X_list)
        B_list = self._estimate_instantaneous_graph(X_list, causal_order)
        A_list = self._estimate_lag_graph(X_list, causal_order, B_list)

        self._causal_order = causal_order
        self._B_list = B_list
        self._A_list = A_list
        self._adjacency_matrices = self._concat_B_A(B_list, A_list, T)

        return self

    def bootstrap(self, X_list, n_sampling):
        """Evaluate the statistical reliability of DAG based on the bootstrapping.

        Parameters
        ----------
        X_list : list of ndarray, shape [(n_samples, n_vars), ...]
            Multiple datasets for training, where each dataset corresponds to a time point.
            The length of X_list is T (number of time points). T >= 2 is required.
        n_sampling : int
            Number of bootstrapping samples.

        Returns
        -------
        results : LongitudinalRESITBootstrapResult
            Returns the results of bootstrapping for multiple datasets.
        """
        # Check parameters
        if not isinstance(X_list, (list, np.ndarray)):
            raise ValueError("X_list must be a array-like.")

        if len(X_list) < 2:
            raise ValueError("X_list must be a list containing at least two items")

        X_list = self._validate_X_list(X_list)
        n_vars = X_list[0].shape[1]
        n_samples = X_list[0].shape[0]
        T = len(X_list)

        causal_orders = []
        B_lists = []
        A_lists = []
        adjacency_matrices_list = []

        # Bootstrapping
        for i in range(n_sampling):
            resampled_X_t = np.empty((T, n_samples, n_vars))
            indices = np.random.randint(0, n_samples, size=(n_samples,))
            for t in range(T):
                resampled_X_t[t] = X_list[t][indices, :]

            self.fit(resampled_X_t)
            causal_orders.append(self.causal_order_)
            B_lists.append(self.B_list_)
            A_lists.append(self.A_list_)
            adjacency_matrices_list.append(self.adjacency_matrices_)

        return LongitudinalRESITBootstrapResult(
            T,
            causal_orders,
            B_lists,
            A_lists,
            np.array(adjacency_matrices_list),
            self._is_common_graph,
        )

    @property
    def causal_order_(self):
        """Instantaneous causal order common to all time points.

        Returns
        -------
        causal_order_ : list of int
            The causal order of the fitted model, where the first element is the most
            upstream variable and the last is the sink.
        """
        if self._causal_order is None:
            raise ValueError("fit() has not been called.")
        return self._causal_order

    @property
    def B_list_(self):
        """Estimated instantaneous causal graphs.

        If is_common_graph=True, a list of length 1 (shared across all time points).
        If is_common_graph=False, a list of length T; B_list_[0], ..., B_list_[n_lags-1] are NaN matrices.
        Each element has shape (n_vars, n_vars).
        B_list_[t][j, i] = 1 means there is an edge from variable i to variable j.

        Returns
        -------
        B_list_ : list of ndarray
        """
        if self._B_list is None:
            raise ValueError("fit() has not been called.")
        return self._B_list

    @property
    def A_list_(self):
        """Estimated lag causal graphs.

        If is_common_graph=True, a list of length 1.
        If is_common_graph=False, a list of length T; A_list_[0], ..., A_list_[n_lags-1] are NaN tensors.
        Each element has shape (n_lags, n_vars, n_vars).
        A_list_[t][k, j, i] = 1 means there is an edge from variable i at k+1 time steps
        before time t to variable j at time t (k is 0-origin).

        Returns
        -------
        A_list_ : list of ndarray
        """
        if self._A_list is None:
            raise ValueError("fit() has not been called.")
        return self._A_list

    @property
    def adjacency_matrices_(self):
        """Estimated adjacency matrices combining instantaneous and lag graphs.

        If is_common_graph=True, a list of length 1.
        If is_common_graph=False, a list of length T; adjacency_matrices_[0], ..., adjacency_matrices_[n_lags-1] are NaN matrices.
        Each element has shape (n_vars, n_vars*(n_lags+1)).
        The first n_vars columns correspond to the instantaneous graph B, and the next n_vars columns correspond to A[0], and so on.

        Returns
        -------
        adjacency_matrices_ : list of ndarray
        """
        if self._adjacency_matrices is None:
            raise ValueError("fit() has not been called.")
        return self._adjacency_matrices

    def _validate_X_list(self, X_list):
        """Check input X list."""
        if not isinstance(X_list, (list, np.ndarray)):
            raise ValueError("X_list must be a array-like.")
        if len(X_list) < 2:
            raise ValueError("X_list must contain at least 2 time points.")

        X_list_ = []
        n_vars = None
        n_samples = None
        for X in X_list:
            X_ = check_array(X)
            if n_vars is None:
                n_vars = X_.shape[1]
            elif X_.shape[1] != n_vars:
                raise ValueError(
                    "All matrices in X_list must have the same number of columns."
                )
            if n_samples is None:
                n_samples = X_.shape[0]
            elif X_.shape[0] != n_samples:
                raise ValueError(
                    "All matrices in X_list must have the same number of rows."
                )
            X_list_.append(X_)

        if len(X_list_) < self._n_lags + 1:
            warnings.warn(
                f"T={len(X_list_)} < n_lags+1={self._n_lags + 1}. "
                "Lag variables are only available for some time points.",
                UserWarning,
                stacklevel=3,
            )

        return X_list_

    def _validate_prior_knowledge(self, n_vars, T):
        """Validate and convert prior_knowledge and prior_knowledge_lag."""
        # prior_knowledge processing
        if self._prior_knowledge_raw is None:
            self._Aknw = None
        else:
            if isinstance(self._prior_knowledge_raw, list):
                raise ValueError(
                    "prior_knowledge must be a single matrix of shape "
                    f"({n_vars}, {n_vars}), not a list."
                )
            arr = check_array(self._prior_knowledge_raw)
            if arr.shape != (n_vars, n_vars):
                raise ValueError(
                    f"prior_knowledge shape {arr.shape} must be ({n_vars}, {n_vars})."
                )
            self._Aknw = np.where(arr < 0, np.nan, arr)

        # Update partial_orders for Phase 1
        if self._Aknw is not None:
            self._partial_orders = self._extract_partial_orders(self._Aknw)
        else:
            self._partial_orders = None

        # prior_knowledge_lag processing
        if self._prior_knowledge_lag_raw is None:
            self._Aknw_lag = None
        else:
            if isinstance(self._prior_knowledge_lag_raw, list):
                raise ValueError(
                    "prior_knowledge_lag must be a single tensor of shape "
                    f"({self._n_lags}, {n_vars}, {n_vars}), not a list."
                )
            arr = np.asarray(self._prior_knowledge_lag_raw, dtype=float)
            if arr.shape != (self._n_lags, n_vars, n_vars):
                raise ValueError(
                    f"prior_knowledge_lag shape {arr.shape} must be "
                    f"({self._n_lags}, {n_vars}, {n_vars})."
                )
            self._Aknw_lag = np.where(arr < 0, np.nan, arr)

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
                    "The prior knowledge contains inconsistencies at the following indices: "
                    f"{pairs[counts > 1].tolist()}"
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
            return check_pairs

        pairs = np.unique(check_pairs, axis=0)
        return pairs[:, [1, 0]]  # [to, from] -> [from, to]

    def _search_candidate(self, S):
        """Search for candidate features"""
        if self._Aknw is None:
            return S
        if self._partial_orders is not None and len(self._partial_orders) != 0:
            Sc = [i for i in S if i not in self._partial_orders[:, 0]]
            return Sc
        return S

    def _apply_prior_knowledge_to_parents(self, j, parents):
        """Filter parent candidates for Phase 2 by removing edges where prior_knowledge=0."""
        if self._Aknw is None:
            return list(parents)
        return [i for i in parents if self._Aknw[j, i] != 0]

    def _apply_prior_knowledge_to_lag_parents(self, j, lag_parents):
        """Filter lag parent candidates for Phase 3 by removing edges where prior_knowledge_lag=0."""
        if self._Aknw_lag is None:
            return list(lag_parents)
        return [(k, i) for (k, i) in lag_parents if self._Aknw_lag[k - 1, j, i] != 0]

    def _compute_residual(self, y, Z):
        """Regress y on Z and return the residuals. Returns y as-is when Z is empty."""
        if Z.shape[1] == 0:
            return y
        self._reg.fit(Z, y)
        return y - self._reg.predict(Z)

    def _fisher_combine_pvalues(self, p_values):
        """Combine a list of p-values using Fisher's method and return the combined p-value."""
        if len(p_values) == 0:
            raise ValueError("p_values must not be empty.")
        chi2_stat = sum(np.inf if p == 0 else -2 * np.log(p) for p in p_values)
        return float(chi2.sf(chi2_stat, df=2 * len(p_values)))

    def _concat_B_A(self, B_list, A_list, T):
        """Concatenate B_list and A_list into a single matrix for counting causal directions."""
        C_list = []
        for B, A in zip(B_list, A_list):
            B = np.asarray(B)
            A = np.asarray(A)
            blocks = [B] + [A[k] for k in range(A.shape[0])]
            C_list.append(np.concatenate(blocks, axis=1))

        if self._is_common_graph:
            C_list = [C_list[0]] * T

        return np.array(C_list)

    # ------------------------------------------------------------------
    # Phase 1: Causal order estimation
    # ------------------------------------------------------------------

    def _estimate_causal_order(self, X_list):
        """Determine topological order"""
        n_vars = X_list[0].shape[1]
        S = list(range(n_vars))
        causal_order = []

        for _ in range(n_vars):
            Sc = self._search_candidate(S)

            if len(Sc) == 1:
                sink = Sc[0]
            else:
                scores = [self._compute_sink_score(X_list, j, S) for j in Sc]
                sink = Sc[int(np.argmax(scores))]

            causal_order.insert(0, sink)
            S.remove(sink)

            if (
                self._Aknw is not None
                and self._partial_orders is not None
                and len(self._partial_orders) > 0
            ):
                self._partial_orders = self._partial_orders[
                    self._partial_orders[:, 1] != sink
                ]

        return causal_order

    def _compute_sink_score(self, X_list, j, S):
        """Compute the sink score (Fisher-combined p-value) for variable j."""
        T = len(X_list)
        p_values = []
        for t in range(1, T):
            n_samples = X_list[t].shape[0]
            parts = []
            same_time_cols = [i for i in S if i != j]
            if same_time_cols:
                parts.append(X_list[t][:, same_time_cols])
            for k in range(1, min(t, self._n_lags) + 1):
                parts.append(X_list[t - k])
            Z = np.hstack(parts) if parts else np.empty((n_samples, 0))
            residual = self._compute_residual(X_list[t][:, j], Z)
            _, p_t = hsic_test_gamma(residual, Z)
            p_values.append(p_t)
        return self._fisher_combine_pvalues(p_values)

    # ------------------------------------------------------------------
    # Phase 2: Instantaneous causal graph estimation
    # ------------------------------------------------------------------

    def _estimate_instantaneous_graph(self, X_list, causal_order):
        """Dispatch Phase 2 processing based on is_common_graph."""
        if self._is_common_graph:
            return self._estimate_common_instantaneous_graph(X_list, causal_order)
        return self._estimate_separate_instantaneous_graph(X_list, causal_order)

    def _estimate_common_instantaneous_graph(self, X_list, causal_order):
        """Estimate the instantaneous causal graph B shared across all time points (Phase 2, is_common_graph=True)."""
        n_vars = X_list[0].shape[1]
        B = np.zeros((n_vars, n_vars))

        for j_idx, j in enumerate(causal_order):
            if j_idx == 0:
                continue

            parents = self._apply_prior_knowledge_to_parents(j, causal_order[:j_idx])

            for i in list(parents):
                p_combined = self._test_edge_combined(X_list, j, i, parents)
                if p_combined < self._alpha:
                    B[j, i] = 1
                else:
                    parents.remove(i)

        return [B]

    def _estimate_separate_instantaneous_graph(self, X_list, causal_order):
        """Estimate independent instantaneous causal graphs per time point (Phase 2, is_common_graph=False)."""
        T = len(X_list)
        n_vars = X_list[0].shape[1]
        B_list = [None] * T
        for t in range(self._n_lags):
            B_list[t] = np.full((n_vars, n_vars), np.nan)

        for t in range(self._n_lags, T):
            B_list[t] = np.zeros((n_vars, n_vars))
            for j_idx, j in enumerate(causal_order):
                if j_idx == 0:
                    continue

                parents_t = self._apply_prior_knowledge_to_parents(
                    j, causal_order[:j_idx]
                )

                for i in list(parents_t):
                    p = self._test_edge_single(X_list, t, j, i, parents_t)
                    if p < self._alpha:
                        B_list[t][j, i] = 1
                    else:
                        parents_t.remove(i)

        return B_list

    def _test_edge_combined(self, X_list, j, i, parents):
        """Return the Fisher-combined p-value across all time points to determine whether to remove edge i->j."""
        T = len(X_list)
        p_values = []
        for t in range(1, T):
            n_samples = X_list[t].shape[0]
            parts_Z = []
            same_time_cols = [p for p in parents if p != i]
            if same_time_cols:
                parts_Z.append(X_list[t][:, same_time_cols])
            for k in range(1, min(t, self._n_lags) + 1):
                parts_Z.append(X_list[t - k])
            Z = np.hstack(parts_Z) if parts_Z else np.empty((n_samples, 0))
            X_parents = X_list[t][:, parents]
            residual = self._compute_residual(X_list[t][:, j], Z)
            _, p_t = hsic_test_gamma(residual, X_parents)
            p_values.append(p_t)
        return self._fisher_combine_pvalues(p_values)

    def _test_edge_single(self, X_list, t, j, i, parents):
        """Return the p-value at time point t alone to determine whether to remove edge i->j."""
        n_samples = X_list[t].shape[0]
        parts_Z = []
        same_time_cols = [p for p in parents if p != i]
        if same_time_cols:
            parts_Z.append(X_list[t][:, same_time_cols])
        for k in range(1, min(t, self._n_lags) + 1):
            parts_Z.append(X_list[t - k])
        Z = np.hstack(parts_Z) if parts_Z else np.empty((n_samples, 0))
        X_parents = X_list[t][:, parents]
        residual = self._compute_residual(X_list[t][:, j], Z)
        _, p = hsic_test_gamma(residual, X_parents)
        return p

    # ------------------------------------------------------------------
    # Phase 3: Lag causal graph estimation
    # ------------------------------------------------------------------

    def _estimate_lag_graph(self, X_list, causal_order, B_list):
        """Dispatch Phase 3 processing based on is_common_graph."""
        if self._is_common_graph:
            return self._estimate_common_lag_graph(X_list, causal_order, B_list[0])
        return self._estimate_separate_lag_graph(X_list, causal_order, B_list)

    def _estimate_common_lag_graph(self, X_list, causal_order, B):
        """Estimate the lag causal graph A shared across all time points (Phase 3, is_common_graph=True)."""
        n_vars = X_list[0].shape[1]
        A = np.zeros((self._n_lags, n_vars, n_vars))

        for j in causal_order:
            confirmed = np.where(B[j, :] == 1)[0].tolist()
            lag_parents = [
                (k, i) for k in range(1, self._n_lags + 1) for i in range(n_vars)
            ]
            lag_parents = self._apply_prior_knowledge_to_lag_parents(j, lag_parents)

            for k_test, i_test in list(lag_parents):
                p_combined = self._test_lag_edge_combined(
                    X_list, j, k_test, i_test, lag_parents, confirmed
                )
                if p_combined < self._alpha:
                    A[k_test - 1, j, i_test] = 1  # k_test: 1-origin -> 0-origin
                else:
                    lag_parents.remove((k_test, i_test))

        return [A]

    def _estimate_separate_lag_graph(self, X_list, causal_order, B_list):
        """Estimate independent lag causal graphs per time point (Phase 3, is_common_graph=False)."""
        T = len(X_list)
        n_vars = X_list[0].shape[1]
        A_list = [None] * T
        for t in range(self._n_lags):
            A_list[t] = np.full((self._n_lags, n_vars, n_vars), np.nan)

        for t in range(self._n_lags, T):
            A_list[t] = np.zeros((self._n_lags, n_vars, n_vars))
            for j in causal_order:
                confirmed = np.where(B_list[t][j, :] == 1)[0].tolist()
                lag_parents = [
                    (k, i) for k in range(1, self._n_lags + 1) for i in range(n_vars)
                ]
                lag_parents = self._apply_prior_knowledge_to_lag_parents(j, lag_parents)

                for k_test, i_test in list(lag_parents):
                    p = self._test_lag_edge_single(
                        X_list, t, j, k_test, i_test, lag_parents, confirmed
                    )
                    if p < self._alpha:
                        A_list[t][k_test - 1, j, i_test] = 1
                    else:
                        lag_parents.remove((k_test, i_test))

        return A_list

    def _test_lag_edge_combined(
        self, X_list, j, k_test, i_test, lag_parents, confirmed_parents
    ):
        """Return the Fisher-combined p-value across all time points for lag edge (k_test, i_test)->j."""
        T = len(X_list)
        p_values = []
        for t in range(1, T):
            if t < k_test:
                continue
            n_samples = X_list[t].shape[0]
            parts_Z = []
            if confirmed_parents:
                parts_Z.append(X_list[t][:, confirmed_parents])
            for k, i in lag_parents:
                if k <= t and (k, i) != (k_test, i_test):
                    parts_Z.append(X_list[t - k][:, i].reshape(-1, 1))
            Z = np.hstack(parts_Z) if parts_Z else np.empty((n_samples, 0))
            lag_cols = []
            for k, i in lag_parents:
                if k <= t:
                    lag_cols.append(X_list[t - k][:, i].reshape(-1, 1))
            X_lag_all = np.hstack(lag_cols) if lag_cols else np.empty((n_samples, 0))
            if X_lag_all.shape[1] == 0:
                continue
            residual = self._compute_residual(X_list[t][:, j], Z)
            _, p = hsic_test_gamma(residual, X_lag_all)
            p_values.append(p)

        if not p_values:
            return 1.0  # No valid time points -> treat as no edge (independent)

        return self._fisher_combine_pvalues(p_values)

    def _test_lag_edge_single(
        self, X_list, t, j, k_test, i_test, lag_parents, confirmed_parents
    ):
        """Return the p-value at time point t alone for lag edge (k_test, i_test)->j."""
        n_samples = X_list[t].shape[0]
        parts_Z = []
        if confirmed_parents:
            parts_Z.append(X_list[t][:, confirmed_parents])
        for k, i in lag_parents:
            if k <= t and (k, i) != (k_test, i_test):
                parts_Z.append(X_list[t - k][:, i].reshape(-1, 1))
        Z = np.hstack(parts_Z) if parts_Z else np.empty((n_samples, 0))
        lag_cols = []
        for k, i in lag_parents:
            if k <= t:
                lag_cols.append(X_list[t - k][:, i].reshape(-1, 1))
        X_lag_all = np.hstack(lag_cols) if lag_cols else np.empty((n_samples, 0))
        residual = self._compute_residual(X_list[t][:, j], Z)
        _, p = hsic_test_gamma(residual, X_lag_all)
        return p


class LongitudinalRESITBootstrapResult(object):
    """The result of bootstrapping for LongitudinalLiNGAM."""

    def __init__(
        self,
        n_timepoints,
        causal_orders,
        B_lists,
        A_lists,
        adjacency_matrices_list,
        is_common_graph,
    ):
        """Construct a BootstrapResult.

        Parameters
        ----------
        n_timepoints : int
            The number of time points.
        causal_orders : array-like, shape (n_sampling)
            The causal order list by bootstrapping.
        B_lists : array-like, shape (n_sampling)
            The instantaneous causal graph list by bootstrapping.
        A_lists : array-like, shape (n_sampling)
            The lag causal graph list by bootstrapping.
        adjacency_matrices_list : array-like, shape (n_sampling)
            The combined adjacency matrices list by bootstrapping.
        is_common_graph : bool
            Whether the estimated graph is common across all time points or separate for each time point.
        """
        self._n_timepoints = n_timepoints
        self._causal_orders = causal_orders
        self._B_lists = B_lists
        self._A_lists = A_lists
        self._adjacency_matrices_list = adjacency_matrices_list
        self._is_common_graph = is_common_graph

    @property
    def causal_orders_(self):
        """Instantaneous causal order common to all time points.

        Returns
        -------
        causal_orders_ : list of list of int
            The causal orders of the fitted model for each bootstrap sample, where the first element is the most
            upstream variable and the last is the sink.
        """
        return self._causal_orders

    @property
    def B_lists_(self):
        """Estimated instantaneous causal graphs.

        Returns
        -------
        B_lists_ : list of ndarray
            The list of estimated instantaneous causal graphs for each bootstrap sample.
        """
        return self._B_lists

    @property
    def A_lists_(self):
        """Estimated lag causal graphs.

        Returns
        -------
        A_lists_ : list of ndarray
            The list of estimated lag causal graphs for each bootstrap sample.
        """
        return self._A_lists

    def get_causal_direction_counts(
        self,
        n_directions=None,
    ):
        """Get causal direction count as a result of bootstrapping.

        Parameters
        ----------
        n_directions : int, optional (default=None)
            If int, then The top ``n_directions`` items are included in the result

        Returns
        -------
        causal_direction_counts : dict
            List of causal directions sorted by count in descending order.
            The dictionary has the following format::

            {'from': [n_directions], 'to': [n_directions], 'count': [n_directions]}

            where ``n_directions`` is the number of causal directions.
        """
        # Check parameters
        if isinstance(n_directions, (numbers.Integral, np.integer)):
            if not 0 < n_directions:
                raise ValueError("n_directions must be an integer greater than 0")
        elif n_directions is None:
            pass
        else:
            raise ValueError("n_directions must be an integer greater than 0")

        if self._is_common_graph:
            T = 1
        else:
            T = self._n_timepoints

        # Count causal directions
        cdc_list = []
        for t in range(T):

            directions = []
            for am in self._adjacency_matrices_list:
                direction = np.array(np.where(np.abs(am[t]) > 0.0))
                directions.append(direction.T)
            directions = np.concatenate(directions)

            if len(directions) == 0:
                cdc = {"from": [], "to": [], "count": []}
                cdc_list.append(cdc)
                continue

            directions, counts = np.unique(directions, axis=0, return_counts=True)
            sort_order = np.argsort(-counts)
            sort_order = (
                sort_order[:n_directions] if n_directions is not None else sort_order
            )
            counts = counts[sort_order]
            directions = directions[sort_order]

            cdc = {
                "from": directions[:, 1].tolist(),
                "to": directions[:, 0].tolist(),
                "count": counts.tolist(),
            }

            cdc_list.append(cdc)

        return cdc_list

    def get_directed_acyclic_graph_counts(self, n_dags=None):
        """Get DAGs count as a result of bootstrapping.

        Parameters
        ----------
        n_dags : int, optional (default=None)
            If int, then The top ``n_dags`` items are included in the result

        Returns
        -------
        directed_acyclic_graph_counts : dict
            List of directed acyclic graphs sorted by count in descending order.
            The dictionary has the following format::

            {'dag': [n_dags], 'count': [n_dags]}.

            where ``n_dags`` is the number of directed acyclic graphs.
        """
        # Check parameters
        if isinstance(n_dags, (numbers.Integral, np.integer)):
            if not 0 < n_dags:
                raise ValueError("n_dags must be an integer greater than 0")
        elif n_dags is None:
            pass
        else:
            raise ValueError("n_dags must be an integer greater than 0")

        if self._is_common_graph:
            T = 1
        else:
            T = self._n_timepoints

        # Count directed acyclic graphs
        dagc_list = []
        for t in range(T):

            dags = []
            for am in self._adjacency_matrices_list:
                dag = np.abs(am[t]) > 0.0
                dags.append(dag)

            dags, counts = np.unique(dags, axis=0, return_counts=True)
            sort_order = np.argsort(-counts)
            sort_order = sort_order[:n_dags] if n_dags is not None else sort_order
            counts = counts[sort_order]
            dags = dags[sort_order]

            dags = [
                {"from": np.where(dag)[1].tolist(), "to": np.where(dag)[0].tolist()}
                for dag in dags
            ]

            dagc_list.append({"dag": dags, "count": counts.tolist()})

        return dagc_list

    def get_probabilities(self):
        """Get bootstrap probability.

        Returns
        -------
        probabilities : array-like
            List of bootstrap probability matrix.
        """
        if self._is_common_graph:
            T = 1
        else:
            T = self._n_timepoints

        # Count directed acyclic graphs
        probs = np.zeros(self._adjacency_matrices_list[0].shape)
        for t in range(T):
            for am in self._adjacency_matrices_list:
                probs[t] += np.where(np.abs(am[t]) > 0.0, 1.0, 0.0)
            probs[t] = probs[t] / len(self._adjacency_matrices_list)

        return probs

    def get_paths(self, from_index, to_index, from_t, to_t):
        """Get all paths from the start variable to the end variable and their bootstrap probabilities.

        Parameters
        ----------
        from_index : int
            Index of the variable at the start of the path.
        to_index : int
            Index of the variable at the end of the path.
        from_t : int
            The starting timepoint of the path.
        to_t : int
            The end timepoint of the path.

        Returns
        -------
        paths : dict
            List of path and bootstrap probability.
            The dictionary has the following format::

            {'path': [n_paths], 'effect': [n_paths], 'probability': [n_paths]}

            where ``n_paths`` is the number of paths.
        """
        # check parameters
        if to_t < from_t:
            raise ValueError("to_t should be greater than or equal to from_t.")
        if to_t == from_t:
            if to_index == from_index:
                raise ValueError("The same variable is specified for from and to.")

        # Find all paths from from_index to to_index
        paths_list = []
        effects_list = []
        for am in self._adjacency_matrices_list:
            # am shape: (T, n_vars, n_vars * (n_lags + 1))
            n_timepoints = to_t - from_t + 1
            n_vars = am.shape[1]
            n_lags = am.shape[2] // n_vars - 1

            expansion_m_size = n_vars * n_timepoints
            expansion_m = np.zeros((expansion_m_size, expansion_m_size))

            # Block(i, j) represents edges from time (from_t + j) to (from_t + i)
            for i in range(n_timepoints):
                t = from_t + i
                row = n_vars * i

                for j in range(i + 1):
                    col = n_vars * j
                    tau = i - j

                    if tau > n_lags:
                        continue

                    c0 = tau * n_vars
                    c1 = (tau + 1) * n_vars

                    expansion_m[row : row + n_vars, col : col + n_vars] = am[
                        t, :, c0:c1
                    ]

                    # row = n_features * i
                    # col = n_features * j
                    # t = i + from_t
                    # tau = i - j
                    # if col > row:
                    #     continue
                    # if tau > am.shape[1] - 1:
                    #     continue
                    # expansion_m[row : row + n_features, col : col + n_features] = am[t, tau]

            paths, effects = find_all_paths(
                expansion_m,
                int(from_index),
                int(n_vars * (to_t - from_t) + to_index),
                0.0,
            )

            # Convert path to string to make them easier to handle.
            paths_list.extend(["_".join(map(str, p)) for p in paths])
            effects_list.extend(effects)

        paths_list = np.array(paths_list)
        effects_list = np.array(effects_list)

        # Count paths
        paths_str, counts = np.unique(paths_list, axis=0, return_counts=True)

        # Sort by count
        order = np.argsort(-counts)
        probs = counts[order] / len(self._adjacency_matrices_list)
        paths_str = paths_str[order]

        # Calculate median of causal effect for each path
        effects = [
            np.median(effects_list[np.where(paths_list == p)]) for p in paths_str
        ]

        result = {
            "path": [[int(i) for i in p.split("_")] for p in paths_str],
            "effect": effects,
            "probability": probs.tolist(),
        }
        return result
