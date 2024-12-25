"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""
import itertools
import numbers
import warnings

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_array

from .direct_lingam import DirectLiNGAM
from .hsic import hsic_test_gamma
from .utils import find_all_paths, calculate_total_effect


class LongitudinalLiNGAM:
    """Implementation of Longitudinal LiNGAM algorithm [1]_

    References
    ----------
    .. [1] K. Kadowaki, S. Shimizu, and T. Washio. Estimation of causal structures
       in longitudinal data using non-Gaussianity. In Proc. 23rd IEEE International
       Workshop on Machine Learning for Signal Processing (MLSP2013), pp. 1--6, Southampton, United Kingdom, 2013.
    """

    def __init__(self, n_lags=1, prior_knowledge=None, measure="pwling", random_state=None):
        """Construct a model.

        Parameters
        ----------
        n_lags : int, optional (default=1)
            Number of lags.
        prior_knowledge : array-like, shape (T, n_lags + 1, n_features, n_features), optional (default=None)
            Prior knowledge used for causal discovery, where ``n_features`` is the number of features.

            The elements of prior knowledge matrix are defined as follows [1]_:

            * ``0`` : :math:`x_i` does not have a directed path to :math:`x_j`
            * ``1`` : :math:`x_i` has a directed path to :math:`x_j`
            * ``-1`` : No prior knowledge is available to know if either of the two cases above (0 or 1) is true.
        measure : {'pwling', 'kernel'}, default='pwling'
            Measure to evaluate independence : 'pwling' or 'kernel'.
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        """
        self._n_lags = n_lags
        self._measure = measure
        self._random_state = random_state
        self._causal_orders = None
        self._adjacency_matrices = None

        if prior_knowledge is not None:
            prior_knowledge = check_array(prior_knowledge, ensure_2d=False, allow_nd=True)
            if len(prior_knowledge.shape) != 4:
                raise ValueError("prior_knowledge must be 4D.")
        self._Aknw = prior_knowledge

    def fit(self, X_list):
        """Fit the model to datasets.

        Parameters
        ----------
        X_list : list, shape [X, ...]
            Longitudinal multiple datasets for training, where ``X`` is an dataset.
            The shape of ``X`` is (n_samples, n_features),
            where ``n_samples`` is the number of samples and ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Check parameters
        if not isinstance(X_list, (list, np.ndarray)):
            raise ValueError("X_list must be a array-like.")

        if len(X_list) < 2:
            raise ValueError("X_list must be a list containing at least two items")

        self._T = len(X_list)
        self._n = check_array(X_list[0]).shape[0]
        self._p = check_array(X_list[0]).shape[1]
        X_t = []
        for X in X_list:
            X = check_array(X)
            if X.shape != (self._n, self._p):
                raise ValueError("X_list must be a list with the same shape")
            X_t.append(X.T)

        n_taus = self._n_lags + 1

        if self._Aknw is None:
            M_tau, N_t = self._compute_residuals(X_t)
            B_t, causal_orders = self._estimate_instantaneous_effects(N_t)
            B_tau = self._estimate_lagged_effects(B_t, M_tau)

            # output B(t,t), B(t,t-τ)
            self._adjacency_matrices = np.empty(
                (self._T, n_taus, self._p, self._p)
            )
            self._adjacency_matrices[:, :] = np.nan
            for t in range(self._n_lags, self._T):
                self._adjacency_matrices[t, 0] = B_t[t]
                for l in range(self._n_lags):
                    if t - l != 0:
                        self._adjacency_matrices[t, l + 1] = B_tau[t, l]

            self._residuals = np.zeros((self._T, self._n, self._p))
            for t in range(self._T):
                self._residuals[t] = N_t[t].T
            self._causal_orders = causal_orders
        else:
            if (self._T, n_taus, self._p, self._p) != self._Aknw.shape:
                raise ValueError(
                    "The shape of prior knowledge must be (T, n_lags + 1, n_features, n_features)"
                )

            X_t = np.vstack(X_t)

            # estimate only instantaneous and lag effects
            pk = np.zeros((self._T * self._p, self._T * self._p))
            for t in range(self._T):
                col_end = (t + 1) * self._p
                col_start = max(col_end - self._p * n_taus, 0)
                pk[
                    t * self._p : (t + 1) * self._p,
                    col_start : col_end
                ] = -1

            # apply the given prior knowledge
            for t in range(self._T):
                for tau in range(n_taus):
                    if t < tau:
                        continue

                    ix = np.ix_(
                        np.arange(
                            t * self._p,
                            (t + 1) * self._p
                        ),
                        np.arange(
                            (t - tau) * self._p,
                            (t - tau + 1) * self._p
                        )
                    )

                    temp = pk[ix]
                    temp[self._Aknw[t, tau] == 0] = 0
                    temp[self._Aknw[t, tau] == 1] = 1
                    pk[ix] = temp

            model = DirectLiNGAM(
                prior_knowledge=pk,
                measure=self._measure,
                random_state=self._random_state
            )
            model.fit(X_t.T)

            # split the estimated adjacency matrix
            adj = np.array(np.split(model.adjacency_matrix_, self._T, axis=1))
            adj = np.array(np.split(adj, self._T, axis=1))

            # construct output matrices
            adjs = np.zeros((self._T, n_taus, self._p, self._p))
            for t in range(self._n_lags, self._T):
                for lag in range(n_taus):
                    adjs[t, lag] = adj[t, t - lag]
            adjs[:self._n_lags] = np.nan
            adjs[:, 1:] = adjs[:, 1:][:, ::-1]

            # make causal_orders
            causal_orders = []
            for t in range(self._T):
                if t < self._n_lags:
                    causal_orders.append([np.nan for _ in range(self._p)])
                    continue

                # extract causal_order at time t
                targets = range(t * self._p, (t + 1) * self._p)
                filter_ = list(map(lambda x: x in targets, model.causal_order_))
                causal_order = np.array(model.causal_order_)[filter_]

                # make numbers start from zero
                causal_order = causal_order - min(causal_order)
                causal_orders.append(causal_order.tolist())

            self._adjacency_matrices = adjs
            self._causal_orders = causal_orders

        return self

    def bootstrap(self, X_list, n_sampling, start_from_t=1):
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
            raise ValueError("X_list must be a array-like.")

        if len(X_list) < 2:
            raise ValueError("X_list must be a list containing at least two items")

        self._T = len(X_list)
        self._n = check_array(X_list[0]).shape[0]
        self._p = check_array(X_list[0]).shape[1]
        X_t = []
        for X in X_list:
            X = check_array(X)
            if X.shape != (self._n, self._p):
                raise ValueError("X_list must be a list with the same shape")
            X_t.append(X)

        # Bootstrapping
        adjacency_matrices = np.zeros(
            (n_sampling, self._T, 1 + self._n_lags, self._p, self._p)
        )
        total_effects = np.zeros((n_sampling, self._T * self._p, self._T * self._p))
        for i in range(n_sampling):
            resampled_X_t = np.empty((self._T, self._n, self._p))
            indices = np.random.randint(0, self._n, size=(self._n,))
            for t in range(self._T):
                resampled_X_t[t] = X_t[t][indices, :]

            self.fit(resampled_X_t)
            adjacency_matrices[i] = self._adjacency_matrices

            # Calculate total effects
            for from_t in range(start_from_t, self._T):
                for c, from_ in enumerate(self._causal_orders[from_t]):
                    to_t = from_t
                    for to in self._causal_orders[from_t][c + 1 :]:
                        total_effects[
                            i, to_t * self._p + to, from_t * self._p + from_
                        ] = self.estimate_total_effect2(from_t, from_, to_t, to)

                    for to_t in range(from_t + 1, self._T):
                        for to in self._causal_orders[to_t]:
                            total_effects[
                                i, to_t * self._p + to, from_t * self._p + from_
                            ] = self.estimate_total_effect2(from_t, from_, to_t, to)

        return LongitudinalBootstrapResult(self._T, adjacency_matrices, total_effects)

    def estimate_total_effect(self, X_t, from_t, from_index, to_t, to_index):
        """Estimate total effect using causal model.

        Parameters
        ----------
        X_t : array-like, shape (timepoint, n_samples, n_features)
            Original data, where n_samples is the number of samples
            and n_features is the number of features.
        from _t :
            The timepoint of source variable.
        from_index :
            Index of source variable to estimate total effect.
        to_t :
            The timepoint of destination variable.
        to_index :
            Index of destination variable to estimate total effect.

        Returns
        -------
        total_effect : float
            Estimated total effect.
        """
        # Check from/to causal order
        if to_t == from_t:
            from_order = self._causal_orders[to_t].index(from_index)
            to_order = self._causal_orders[from_t].index(to_index)
            if from_order > to_order:
                warnings.warn(
                    f"The estimated causal effect may be incorrect because "
                    f"the causal order of the destination variable (to_t={to_t}, to_index={to_index}) "
                    f"is earlier than the source variable (from_t={from_t}, from_index={from_index})."
                )
        elif to_t < from_t:
            warnings.warn(
                f"The estimated causal effect may be incorrect because "
                f"the causal order of the destination variable (to_t={to_t}) "
                f"is earlier than the source variable (from_t={from_t})."
            )

        # X + lagged X
        # n_features * (to + from + n_lags)
        X_joined = np.zeros((self._n, self._p * (2 + self._n_lags)))
        X_joined[:, 0 : self._p] = X_t[to_t]
        for tau in range(1 + self._n_lags):
            pos = self._p + self._p * tau
            X_joined[:, pos : pos + self._p] = X_t[from_t - tau]

        am = np.concatenate([*self._adjacency_matrices[from_t]], axis=1)

        # from_index + parents indices
        parents = np.where(np.abs(am[from_index]) > 0)[0]
        predictors = [from_index + self._p]
        predictors.extend(parents + self._p)

        # Estimate total effect
        lr = LinearRegression()
        lr.fit(X_joined[:, predictors], X_joined[:, to_index])

        return lr.coef_[0]

    def estimate_total_effect2(self, from_t, from_index, to_t, to_index):
        """Estimate total effect using causal model.

        Parameters
        ----------
        from _t :
            The timepoint of source variable.
        from_index :
            Index of source variable to estimate total effect.
        to_t :
            The timepoint of destination variable.
        to_index :
            Index of destination variable to estimate total effect.

        Returns
        -------
        total_effect : float
            Estimated total effect.
        """
        # Check from/to causal order
        if to_t == from_t:
            from_order = self._causal_orders[to_t].index(from_index)
            to_order = self._causal_orders[from_t].index(to_index)
            if from_order > to_order:
                warnings.warn(
                    f"The estimated causal effect may be incorrect because "
                    f"the causal order of the destination variable (to_t={to_t}, to_index={to_index}) "
                    f"is earlier than the source variable (from_t={from_t}, from_index={from_index})."
                )
        elif to_t < from_t:
            warnings.warn(
                f"The estimated causal effect may be incorrect because "
                f"the causal order of the destination variable (to_t={to_t}) "
                f"is earlier than the source variable (from_t={from_t})."
            )

        n_timepoints = to_t - from_t + 1
        n_features = self._adjacency_matrices.shape[2]
        expansion_m_size = n_features * n_timepoints
        expansion_m = np.zeros((expansion_m_size, expansion_m_size))
        for i in range(n_timepoints):
            for j in range(n_timepoints):
                row = n_features * i
                col = n_features * j
                t = i + from_t
                tau = i - j
                if col > row:
                    continue
                if tau > self._n_lags:
                    continue
                expansion_m[row : row + n_features, col : col + n_features] = self._adjacency_matrices[t, tau]

        effect = calculate_total_effect(expansion_m, from_index, n_features * (to_t - from_t) + to_index)

        return effect

    def get_error_independence_p_values(self):
        """Calculate the p-value matrix of independence between error variables.

        Returns
        -------
        independence_p_values : array-like, shape (n_features, n_features)
            p-value matrix of independence between error variables.
        """
        E_list = np.empty((self._T, self._n, self._p))
        for t, resid in enumerate(self.residuals_):
            B_t = self._adjacency_matrices[t, 0]
            E_list[t] = np.dot(np.eye(B_t.shape[0]) - B_t, resid.T).T

        p_values_list = np.zeros([self._T, self._p, self._p])
        p_values_list[:, :, :] = np.nan
        for t in range(1, self._T):
            p_values = np.zeros([self._p, self._p])
            for i, j in itertools.combinations(range(self._p), 2):
                _, p_value = hsic_test_gamma(
                    np.reshape(E_list[t][:, i], [self._n, 1]),
                    np.reshape(E_list[t][:, j], [self._n, 1]),
                )
                p_values[i, j] = p_value
                p_values[j, i] = p_value

            p_values_list[t] = p_values

        return p_values_list

    def _compute_residuals(self, X_t):
        """Compute residuals N(t)"""
        M_tau = np.zeros((self._T, self._n_lags, self._p, self._p))
        N_t = np.zeros((self._T, self._p, self._n))
        N_t[:, :, :] = np.nan

        X_t = np.array(X_t)

        for t in range(self._n_lags, self._T):
            X = np.vstack(X_t[t - self._n_lags:t])
            y = X_t[t]

            reg = LinearRegression()
            reg.fit(X.T, y.T)

            M_tau[t, :self._n_lags] = np.split(reg.coef_, self._n_lags, axis=1)
            N_t[t] = X_t[t] - reg.coef_ @ X

        return M_tau, N_t

    def _estimate_instantaneous_effects(self, N_t):
        """Estimate instantaneous effects B(t,t) by applying LiNGAM"""
        causal_orders = [[np.nan] * self._p for _ in range(self._n_lags)]
        B_t = np.zeros((self._T, self._p, self._p))
        for t in range(self._n_lags, self._T):
            model = DirectLiNGAM(measure=self._measure)
            model.fit(N_t[t].T)
            causal_orders.append(model.causal_order_)
            B_t[t] = model.adjacency_matrix_
        return B_t, causal_orders

    def _estimate_lagged_effects(self, B_t, M_tau):
        """Estimate lagged effects B(t,t-τ)"""
        B_tau = np.zeros((self._T, self._n_lags, self._p, self._p))
        for t in range(self._n_lags, self._T):
            for tau in range(self._n_lags):
                B_tau[t, tau] = np.dot(np.eye(self._p) - B_t[t], M_tau[t, tau])
        return B_tau

    @property
    def causal_orders_(self):
        """Estimated causal ordering.

        Returns
        -------
        causal_order_ : array-like, shape (causal_order, ...)
            The causal order of fitted models for B(t,t).
            The shape of causal_order is (n_features),
            where ``n_features`` is the number of features.
        """
        return self._causal_orders

    @property
    def adjacency_matrices_(self):
        """Estimated adjacency matrices.

        Returns
        -------
        adjacency_matrices_ : array-like, shape ((B(t,t), B(t,t-1), ..., B(t,t-τ)), ...)
            The list of adjacency matrix B(t,t) and B(t,t-τ) for longitudinal datasets.
            The shape of B(t,t) and B(t,t-τ) is (n_features, n_features), where
            ``n_features`` is the number of features.
            **If the previous data required for the calculation are not available,
            such as B(t,t) or B(t,t-τ) at t=0, all elements of the matrix are nan**.
        """
        return self._adjacency_matrices

    @property
    def residuals_(self):
        """Residuals of regression.

        Returns
        -------
        residuals_ : list, shape [E, ...]
            Residuals of regression, where ``E`` is an dataset.
            The shape of ``E`` is (n_samples, n_features),
            where ``n_samples`` is the number of samples and ``n_features`` is the number of features.

        """
        return self._residuals


class LongitudinalBootstrapResult(object):
    """The result of bootstrapping for LongitudinalLiNGAM."""

    def __init__(self, n_timepoints, adjacency_matrices, total_effects):
        """Construct a BootstrapResult.

        Parameters
        ----------
        adjacency_matrices : array-like, shape (n_sampling)
            The adjacency matrix list by bootstrapping.
        total_effects : array-like, shape (n_sampling)
            The total effects list by bootstrapping.
        """
        self._n_timepoints = n_timepoints
        self._adjacency_matrices = adjacency_matrices
        self._total_effects = total_effects

    @property
    def adjacency_matrices_(self):
        """The adjacency matrix list by bootstrapping.

        Returns
        -------
        adjacency_matrices_ : array-like, shape (n_sampling)
            The adjacency matrix list, where ``n_sampling`` is
            the number of bootstrap sampling.
        """
        return self._adjacency_matrices

    @property
    def total_effects_(self):
        """The total effect list by bootstrapping.

        Returns
        -------
        total_effects_ : array-like, shape (n_sampling)
            The total effect list, where ``n_sampling`` is
            the number of bootstrap sampling.
        """
        return self._total_effects

    def get_causal_direction_counts(
        self,
        n_directions=None,
        min_causal_effect=None,
        split_by_causal_effect_sign=False,
    ):
        """Get causal direction count as a result of bootstrapping.

        Parameters
        ----------
        n_directions : int, optional (default=None)
            If int, then The top ``n_directions`` items are included in the result
        min_causal_effect : float, optional (default=None)
            Threshold for detecting causal direction.
            If float, then causal directions with absolute values of causal
            effects less than ``min_causal_effect`` are excluded.
        split_by_causal_effect_sign : boolean, optional (default=False)
            If True, then causal directions are split depending on the sign of the causal effect.

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

        if min_causal_effect is None:
            min_causal_effect = 0.0
        else:
            if not 0.0 < min_causal_effect:
                raise ValueError("min_causal_effect must be an value greater than 0.")

        # Count causal directions
        cdc_list = []
        for t in range(self._n_timepoints):

            directions = []
            for m in self._adjacency_matrices:
                am = np.concatenate([*m[t]], axis=1)
                direction = np.array(np.where(np.abs(am) > min_causal_effect))
                if split_by_causal_effect_sign:
                    signs = (
                        np.array([np.sign(am[i][j]) for i, j in direction.T])
                        .astype("int64")
                        .T
                    )
                    direction = np.vstack([direction, signs])
                directions.append(direction.T)
            directions = np.concatenate(directions)

            if len(directions) == 0:
                cdc = {"from": [], "to": [], "count": []}
                if split_by_causal_effect_sign:
                    cdc["sign"] = []
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
            if split_by_causal_effect_sign:
                cdc["sign"] = directions[:, 2].tolist()

            cdc_list.append(cdc)

        return cdc_list

    def get_directed_acyclic_graph_counts(
        self, n_dags=None, min_causal_effect=None, split_by_causal_effect_sign=False
    ):
        """Get DAGs count as a result of bootstrapping.

        Parameters
        ----------
        n_dags : int, optional (default=None)
            If int, then The top ``n_dags`` items are included in the result
        min_causal_effect : float, optional (default=None)
            Threshold for detecting causal direction.
            If float, then causal directions with absolute values of causal effects less than
            ``min_causal_effect`` are excluded.
        split_by_causal_effect_sign : boolean, optional (default=False)
            If True, then causal directions are split depending on the sign of the causal effect.

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

        if min_causal_effect is None:
            min_causal_effect = 0.0
        else:
            if not 0.0 < min_causal_effect:
                raise ValueError("min_causal_effect must be an value greater than 0.")

        # Count directed acyclic graphs
        dagc_list = []
        for t in range(self._n_timepoints):

            dags = []
            for m in self._adjacency_matrices:
                am = np.concatenate([*m[t]], axis=1)

                dag = np.abs(am) > min_causal_effect
                if split_by_causal_effect_sign:
                    direction = np.array(np.where(dag))
                    signs = np.zeros_like(dag).astype("int64")
                    for i, j in direction.T:
                        signs[i][j] = np.sign(am[i][j]).astype("int64")
                    dag = signs
                dags.append(dag)

            dags, counts = np.unique(dags, axis=0, return_counts=True)
            sort_order = np.argsort(-counts)
            sort_order = sort_order[:n_dags] if n_dags is not None else sort_order
            counts = counts[sort_order]
            dags = dags[sort_order]

            if split_by_causal_effect_sign:
                dags = [
                    {
                        "from": np.where(dag)[1].tolist(),
                        "to": np.where(dag)[0].tolist(),
                        "sign": [dag[i][j] for i, j in np.array(np.where(dag)).T],
                    }
                    for dag in dags
                ]
            else:
                dags = [
                    {"from": np.where(dag)[1].tolist(), "to": np.where(dag)[0].tolist()}
                    for dag in dags
                ]

            dagc_list.append({"dag": dags, "count": counts.tolist()})

        return dagc_list

    def get_probabilities(self, min_causal_effect=None):
        """Get bootstrap probability.

        Parameters
        ----------
        min_causal_effect : float, optional (default=None)
            Threshold for detecting causal direction.
            If float, then causal directions with absolute values of causal effects less than
            ``min_causal_effect`` are excluded.

        Returns
        -------
        probabilities : array-like
            List of bootstrap probability matrix.
        """
        # check parameters
        if min_causal_effect is None:
            min_causal_effect = 0.0
        else:
            if not 0.0 < min_causal_effect:
                raise ValueError("min_causal_effect must be an value greater than 0.")

        prob = np.zeros(self._adjacency_matrices[0].shape)
        for adj_mat in self._adjacency_matrices:
            prob += np.where(np.abs(adj_mat) > min_causal_effect, 1, 0)
        prob = prob / len(self._adjacency_matrices)

        return prob

    def get_total_causal_effects(self, min_causal_effect=None):
        """Get total effects list.

        Parameters
        ----------
        min_causal_effect : float, optional (default=None)
            Threshold for detecting causal direction.
            If float, then causal directions with absolute values of causal effects less than
            ``min_causal_effect`` are excluded.

        Returns
        -------
        total_causal_effects : dict
            List of bootstrap total causal effect sorted by probability in descending order.
            The dictionary has the following format::

            {'from': [n_directions], 'to': [n_directions], 'effect': [n_directions], 'probability': [n_directions]}

            where ``n_directions`` is the number of causal directions.
        """
        # Check parameters
        if min_causal_effect is None:
            min_causal_effect = 0.0
        else:
            if not 0.0 < min_causal_effect:
                raise ValueError("min_causal_effect must be an value greater than 0.")

        # probability
        probs = np.sum(
            np.where(np.abs(self._total_effects) > min_causal_effect, 1, 0),
            axis=0,
            keepdims=True,
        )[0]
        probs = probs / len(self._total_effects)

        # causal directions
        dirs = np.array(np.where(np.abs(probs) > 0))
        probs = probs[dirs[0], dirs[1]]

        # calculate median effect without zero
        effects = np.zeros(dirs.shape[1])
        for i, (to, from_) in enumerate(dirs.T):
            idx = np.where(np.abs(self._total_effects[:, to, from_]) > 0)
            effects[i] = np.median(self._total_effects[:, to, from_][idx])

        # sort by effect value
        order = np.argsort(-probs)
        dirs = dirs.T[order]
        effects = effects[order]
        probs = probs[order]

        ce = {
            "from": dirs[:, 1].tolist(),
            "to": dirs[:, 0].tolist(),
            "effect": effects.tolist(),
            "probability": probs.tolist(),
        }

        return ce

    def get_paths(self, from_index, to_index, from_t, to_t, min_causal_effect=None):
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
        min_causal_effect : float, optional (default=None)
            Threshold for detecting causal direction.
            Causal directions with absolute values of causal effects less than ``min_causal_effect`` are excluded.

        Returns
        -------
        paths : dict
            List of path and bootstrap probability.
            The dictionary has the following format::

            {'path': [n_paths], 'effect': [n_paths], 'probability': [n_paths]}

            where ``n_paths`` is the number of paths.
        """
        # check parameters
        if min_causal_effect is None:
            min_causal_effect = 0.0
        else:
            if not 0.0 < min_causal_effect:
                raise ValueError("min_causal_effect must be an value greater than 0.")
        if to_t < from_t:
            raise ValueError("to_t should be greater than or equal to from_t.")
        if to_t == from_t:
            if to_index == from_index:
                raise ValueError("The same variable is specified for from and to.")

        # Find all paths from from_index to to_index
        paths_list = []
        effects_list = []
        for am in self._adjacency_matrices:
            n_timepoints = to_t - from_t + 1
            n_features = am.shape[2]
            expansion_m_size = n_features * n_timepoints
            expansion_m = np.zeros((expansion_m_size, expansion_m_size))
            for i in range(n_timepoints):
                for j in range(n_timepoints):
                    row = n_features * i
                    col = n_features * j
                    t = i + from_t
                    tau = i - j
                    if col > row:
                        continue
                    if tau > am.shape[1] - 1:
                        continue
                    expansion_m[row : row + n_features, col : col + n_features] = am[t, tau]

            paths, effects = find_all_paths(
                expansion_m,
                int(from_index),
                int(n_features * (to_t - from_t) + to_index),
                min_causal_effect,
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
        probs = counts[order] / len(self._adjacency_matrices)
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
