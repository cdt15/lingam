"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""

import numpy as np
from sklearn.utils import check_array

from .base import _BaseLiNGAM
from .hsic import hsic_test_gamma


class RESIT(_BaseLiNGAM):
    """Implementation of RESIT(regression with subsequent independence test) Algorithm [1]_

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

    def __init__(self, regressor, random_state=None, prior_knowledge=None, alpha=0.01):
        """Construct a RESIT model.

        Parameters
        ----------
        regressor : regressor object implementing 'fit' and 'predict' function (default=None)
            Regressor to compute residuals.
            This regressor object must have ``fit`` method and ``predict`` function like scikit-learn's model.
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        prior_knowledge : array-like, shape (n_features, n_features), optional (default=None)
            Prior knowledge used for causal discovery, where ``n_features`` is the number of features.

            The elements of prior knowledge matrix are defined as follows [1]_:

            * ``0`` : :math:`x_i` does not have a directed path to :math:`x_j`
            * ``1`` : :math:`x_i` has a directed path to :math:`x_j`
            * ``-1`` : No prior knowledge is available to know if either of the two cases above (0 or 1) is true.
        alpha : float, optional (default=0.01)
            Alpha level for HSIC independence test when removing superfluous edges.
        """
        # Check parameters
        if regressor is None:
            raise ValueError("Specify regression model in 'regressor'.")
        else:
            if not (hasattr(regressor, "fit") and hasattr(regressor, "predict")):
                raise ValueError("'regressor' has no fit or predict method.")

        if alpha < 0.0:
            raise ValueError("alpha must be an float greater than 0.")

        super().__init__(random_state)
        self._Aknw = prior_knowledge
        self._alpha = alpha
        self._reg = regressor

        if self._Aknw is not None:
            self._Aknw = check_array(self._Aknw)
            self._Aknw = np.where(self._Aknw < 0, np.nan, self._Aknw)

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
        # Check parameters
        X = check_array(X)
        n_features = X.shape[1]

        # Check prior knowledge
        if self._Aknw is not None:
            if (n_features, n_features) != self._Aknw.shape:
                raise ValueError(
                    "The shape of prior knowledge must be (n_features, n_features)"
                )
            else:
                # Extract all partial orders in prior knowledge matrix
                self._partial_orders = self._extract_partial_orders(self._Aknw)

        # Determine topological order
        pa, pi = self._estimate_order(X)

        # Remove superfluous edges
        pa = self._remove_edges(X, pa, pi)

        # Create adjacency matrix from parent-child relationship
        adjacency_matrix = np.zeros([n_features, n_features])
        for i, parents in pa.items():
            for p in parents:
                adjacency_matrix[i, p] = 1

        self._causal_order = pi
        self._adjacency_matrix = adjacency_matrix
        return self

    def _estimate_order(self, X):
        """Determine topological order"""
        S = np.arange(X.shape[1])
        pa = {}
        pi = []

        for _ in range(X.shape[1]):
            Sc = self._search_candidate(S)

            if len(Sc)==1:
                k = Sc[0]
            else:
                hsic_stats = []
                for k in Sc:
                    # Regress Xk on {Xi}
                    predictors = [i for i in S if i != k]
                    self._reg.fit(X[:, predictors], X[:, k])
                    residual = X[:, k] - self._reg.predict(X[:, predictors])

                    # Measure dependence between residuals and {Xi}
                    hsic_stat, _ = hsic_test_gamma(residual, X[:, predictors])
                    hsic_stats.append(hsic_stat)

                k = Sc[np.argmin(hsic_stats)]

            S = S[S != k]
            pa[k] = S.tolist()
            pi.insert(0, k)

            # Update partial orders
            if self._Aknw is not None:
                self._partial_orders = self._partial_orders[self._partial_orders[:, 1] != k]

        # The topological order after exploration is revised based on prior knowledge.
        if self._Aknw is not None:
            _pk = self._Aknw.copy()
            np.fill_diagonal(_pk, 0)

            pk_pa = {}
            for i, parents in pa.items():
                new_parents = [p for p in parents if _pk[i, p] != 0]
                pk_pa[i] = new_parents
            pa = pk_pa

        return pa, pi

    def _remove_edges(self, X, pa, pi):
        """Remove superfluous edges"""
        for k in range(1, X.shape[1]):
            parents = pa[pi[k]].copy()

            for l in parents:
                # Regress Xk on {Xi}
                predictors = [i for i in pa[pi[k]] if i != l]
                if len(predictors) >= 1:
                    self._reg.fit(X[:, predictors], X[:, pi[k]])
                    residual = X[:, pi[k]] - self._reg.predict(X[:, predictors])
                else:
                    residual = X[:, pi[k]]
                # Measure dependence between residuals and {Xi}
                _, hsic_p = hsic_test_gamma(residual, X[:, parents])
                # eliminate edge
                if hsic_p > self._alpha:
                    pa[pi[k]].remove(l)

        return pa

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
            Sc = [i for i in S if i not in self._partial_orders[:, 0]]
            return Sc

        return S

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
            **Because RESIT is a nonlinear algorithm, it cannot estimate the total effect and always returns a value of zero**
        """
        return 0

    def get_error_independence_p_values(self, X):
        """Calculate the p-value matrix of independence between error variables.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Original data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        independence_p_values : array-like, shape (n_features, n_features)
            **RESIT always returns zero**
        """
        n_features = X.shape[1]
        p_values = np.zeros([n_features, n_features])
        return p_values
