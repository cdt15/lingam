"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""

import numpy as np
import statsmodels.api as sm
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, LassoLarsIC

from .base import _BaseLiNGAM
from .utils import bic_select_logistic_l1


class mLiNGAM(_BaseLiNGAM):
    """Implementation of the Missingness-LiNGAM (m-LiNGAM) Algorithm
    """
    def __init__(
        self,
        random_state=None,
        prior_knowledge=None,
        apply_prior_knowledge_softly=False,
        missingness_knowledge=None,
    ):
        """Construct an m-LiNGAM model.

        Parameters
        ----------
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        prior_knowledge : array-like, shape (n_features, n_features), optional (default=None)
            Prior knowledge used for causal discovery, where ``n_features`` is the number of features.

            The elements of prior knowledge matrix are defined as follows [1]_:

            * ``0`` : :math:`x_i` does not have a directed path to :math:`x_j`
            * ``1`` : :math:`x_i` has a directed path to :math:`x_j`
            * ``-1`` : No prior knowledge is available to know if either of the two cases above (0 or 1) is true.
        apply_prior_knowledge_softly : boolean, optional (default=False)
            If True, apply prior knowledge softly.
        """
        super().__init__(random_state)
        self._Aknw = prior_knowledge
        self._apply_prior_knowledge_softly = apply_prior_knowledge_softly
        self._missingness_knowledge=missingness_knowledge
        self._missingness_mechanisms_parents = {}  # Dictionary. Key: partially observed variable index, Value: corresponding missingness mechanism's parents indexes (list) 
        self._missingness_mechanisms_coef = {}
        self.n_features = None
        self._descendants_mechanisms = None
        self._missing_to_index = None
        self._intercept = None

        if self._Aknw is not None:
            self._Aknw = check_array(self._Aknw)
            self._Aknw = np.where(self._Aknw < 0, np.nan, self._Aknw)

    def fit(self, X):
        """Fit the model to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features. Missing values may be represented as NaN.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Check parameters
        X = check_array(X, ensure_all_finite='allow-nan')
        self.n_features = X.shape[1]

        # Check prior knowledge
        if self._Aknw is not None:
            if (self.n_features, self.n_features) != self._Aknw.shape:
                raise ValueError(
                    "The shape of prior knowledge must be (n_features, n_features)"
                )
            else:
                # Extract all partial orders in prior knowledge matrix
                if not self._apply_prior_knowledge_softly:
                    self._partial_orders = self._extract_partial_orders(self._Aknw)

        
        # Causal discovery
        U = np.arange(self.n_features)
        K = []
        X_ = np.copy(X)

        # Discover missingness mechanisms
        missing_mask = np.isnan(X_)
        columns_with_missing = np.any(missing_mask, axis=0)
        missing_column_indices = np.where(columns_with_missing)[0]
        self._missing_to_index = {i:m for m,i in enumerate(missing_column_indices)}
        self._descendants_mechanisms = -1*np.ones([len(missing_column_indices), self.n_features])

        R = {i:missing_mask[:,i] for i in missing_column_indices}
        for k in R.keys():
            # Find the parent nodes of the missingness mechanism
            available_rows = ~np.any(np.isnan(np.delete(X_, k, axis=1)), axis=1)

            X_lreg = np.delete(X_, k, axis=1)[available_rows]
            scaler = StandardScaler()
            X_lreg = scaler.fit_transform(X_lreg)

            best_coef, _, _, _, _ = bic_select_logistic_l1(X_lreg, R[k][available_rows], Cs=50, max_iter=1000)
            
            self._missingness_mechanisms_parents[k] = list(np.arange(self.n_features)[np.where(best_coef!=0)])
            self._missingness_mechanisms_parents[k] = [idx if idx<k else idx+1 for idx in self._missingness_mechanisms_parents[k]]
            available_rows = ~np.any(np.isnan(X_[:,self._missingness_mechanisms_parents[k]]), axis=1)

            if len(self._missingness_mechanisms_parents[k])==0:
                clf = LogisticRegression(
                    penalty=None,
                    solver='lbfgs',
                    max_iter=1000,
                    fit_intercept=False
                )
                #independent_vars = np.ones_like(R[k])
                independent_vars = np.ones_like(R[k]).reshape(-1, 1)
                clf.fit(independent_vars, R[k][available_rows])
                self._missingness_mechanisms_coef[k] = np.concatenate([clf.coef_.ravel()])
            else:
                clf = LogisticRegression(
                    penalty='l2',
                    C=0.5,
                    solver='lbfgs',
                    max_iter=1000,
                    fit_intercept=True
                )
                independent_vars = X_[:, self._missingness_mechanisms_parents[k]][available_rows]
                clf.fit(independent_vars, R[k][available_rows])
                self._missingness_mechanisms_coef[k] = np.concatenate([clf.intercept_, clf.coef_.ravel()])
            
        # Estimate causal order
        X_top = X_.copy()

        for _ in range(len(U)):
            m = self._search_causal_order_top_down(X_top, U, min_samples=self.n_features+1)
            for i in U:
                if i != m:
                    X_top[:, i][~np.isnan(X[:,[i,m]]).any(axis=1)] = self._residual(X_top[:, i][~np.isnan(X[:,[i,m]]).any(axis=1)], X_top[:, m][~np.isnan(X[:,[i,m]]).any(axis=1)])
            K.append(m)
            U = U[U != m]
            # Update partial orders
            if (self._Aknw is not None) and (not self._apply_prior_knowledge_softly):
                self._partial_orders = self._partial_orders[
                    self._partial_orders[:, 0] != m
                ]

        self._causal_order = K
        
        # Estimate adjacency matrix
        if self._Aknw is not None:
            pk = self._Aknw.copy()
            np.fill_diagonal(pk, 0)

        B = np.zeros([X.shape[1], X.shape[1]], dtype="float64")
        c = np.zeros(X.shape[1], dtype="float64")
        for i in range(len(self._causal_order)):
            target = self._causal_order[i]
            predictors = self._causal_order[:i]

            # Exclude variables specified in no_path with prior knowledge
            if self._Aknw is not None:
                predictors = [p for p in predictors if pk[target, p] != 0]

            while len(predictors)>0:
                n_pred=len(predictors)
                # Calculate coefficients of the original scale
                involved_variables = set([target]+predictors)
                num_involved = len(involved_variables)
                while(True):
                    for j in involved_variables:
                        if j in self._missingness_mechanisms_parents.keys():
                            involved_variables = involved_variables.union(set(self._missingness_mechanisms_parents[j]))
                    if num_involved == len(involved_variables):
                        break
                    else:
                        num_involved = len(involved_variables)
                X_m = X[~np.any(np.isnan(X[:, sorted(involved_variables)]), axis=1)]
                sample_weight = np.prod([1/(1-self._logistic_prediction(X_m[:,self._missingness_mechanisms_parents[i]], self._missingness_mechanisms_coef[i])) for i in [target]+predictors if i in self._missingness_mechanisms_parents.keys()], axis=0)

                # Standardize X
                scaler = StandardScaler()
                X_std = scaler.fit_transform(X_m)

                # Pruning with Adaptive Lasso
                lr = Lasso(alpha=0.1)
                lr.fit(X_std[:, predictors], X_std[:, target], sample_weight=sample_weight)
                weight = np.power(np.abs(lr.coef_), 1.0)
                reg = LassoLarsIC(criterion="bic")
                reg.fit(X_std[:, predictors] * weight, X_std[:, target])
                pruned_idx = np.abs(reg.coef_ * weight) > 0.0

                pred = np.array(predictors)
                predictors = list(pred[pruned_idx])

                if len(predictors)==n_pred:
                    break

            # Calculate coefficients of the original scale
            involved_variables = set([target]+predictors)
            num_involved = len(involved_variables)
            while(True):
                for j in involved_variables:
                    if j in self._missingness_mechanisms_parents.keys():
                        involved_variables = involved_variables.union(set(self._missingness_mechanisms_parents[j]))
                if num_involved == len(involved_variables):
                    break
                else:
                    num_involved = len(involved_variables)
            X_m = X[~np.any(np.isnan(X[:, sorted(involved_variables)]), axis=1)]
            sample_weight = np.prod([1/(1-self._logistic_prediction(X_m[:,self._missingness_mechanisms_parents[i]], self._missingness_mechanisms_coef[i])) for i in [target]+predictors if i in self._missingness_mechanisms_parents.keys()], axis=0)
            
            if len(predictors)==0:
                X_design = np.ones((X_m.shape[0], 1))
                lr = LinearRegression(fit_intercept=False)
                lr.fit(X_design, X_m[:, target], sample_weight=sample_weight)
                c[target] = lr.coef_[0]
            else:
                coef = np.zeros(reg.coef_.shape)
                lr = LinearRegression(fit_intercept=True)
                lr.fit(X_m[:, pred[pruned_idx]], X_m[:, target], sample_weight=sample_weight)
                coef[pruned_idx] = lr.coef_
                B[target, predictors] = coef
                c[target] = lr.intercept_
        
        self._adjacency_matrix = B
        self._intercept = c

        return self


    def _logistic_prediction(self, data, params):
        return 1 / (1 + np.exp(-np.dot(sm.add_constant(data), params)))

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

        # Check for inconsistencies in pairs without path.
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

    def _residual(self, xi, xj):
        """The residual when xi is regressed on xj."""
        return xi - (np.cov(xi, xj, bias=True)[0, 1] / np.var(xj)) * xj

    def _entropy(self, u):
        """Calculate entropy using the maximum entropy approximations."""
        k1 = 79.047
        k2 = 7.4129
        gamma = 0.37457
        return (1 + np.log(2 * np.pi)) / 2 - k1 * (
            np.mean(np.log(np.cosh(u))) - gamma) ** 2 - k2 * (np.mean(u * np.exp((-(u ** 2)) / 2))) ** 2

    def _diff_mutual_info(self, xi_std, xj_std, ri_j, rj_i):
        """Calculate the difference of the mutual informations."""
        return (self._entropy(xj_std) + self._entropy(ri_j / np.std(ri_j))) - (
            self._entropy(xi_std) + self._entropy(rj_i / np.std(rj_i))
        )

    def _search_candidate_top_down(self, U):
        """Search for candidate features"""
        # If no prior knowledge is specified, nothing to do.
        if self._Aknw is None:
            return U, []

        # Apply prior knowledge in a strong way
        if not self._apply_prior_knowledge_softly:
            if len(self._partial_orders) != 0:
                Uc = [i for i in U if i not in self._partial_orders[:, 1]]
                return Uc, []
            else:
                return U, []

        # Find exogenous features
        Uc = []
        for j in U:
            index = U[U != j]
            if self._Aknw[j][index].sum() == 0:
                Uc.append(j)

        # Find endogenous features, and then find candidate features
        if len(Uc) == 0:
            U_end = []
            for j in U:
                index = U[U != j]
                if np.nansum(self._Aknw[j][index]) > 0:
                    U_end.append(j)

            # Find sink features
            for i in U:
                index = U[U != i]
                if self._Aknw[index, i].sum() == 0:
                    U_end.append(i)
            Uc = [i for i in U if i not in set(U_end)]

        # make V^(j)
        Vj = []
        for i in U:
            if i in Uc:
                continue
            if self._Aknw[i][Uc].sum() == 0:
                Vj.append(i)
        return Uc, Vj

    def _search_causal_order_top_down(self, X, U, min_samples=0):
        """Search the causal ordering from top to bottom."""
        Uc, Vj = self._search_candidate_top_down(U)
        if len(Uc) == 1:
            return Uc[0]

        M_list = []
        for i in Uc:
            M = 0
            for j in U:
                if i != j:
                    X_m = X[:,[i,j]].copy()
                    X_m = X_m[~np.isnan(X_m).any(axis=1)]
                    if X_m.shape[0]<min_samples:
                        return -1
                    xi_std = (X_m[:, 0] - np.mean(X_m[:, 0])) / np.std(X_m[:, 0])
                    xj_std = (X_m[:, 1] - np.mean(X_m[:, 1])) / np.std(X_m[:, 1])
                    ri_j = (
                        xi_std
                        if i in Vj and j in Uc
                        else self._residual(xi_std, xj_std)
                    )
                    rj_i = (
                        xj_std
                        if j in Vj and i in Uc
                        else self._residual(xj_std, xi_std)
                    )
                    M += np.min([0, self._diff_mutual_info(xi_std, xj_std, ri_j, rj_i)]) ** 2
            M_list.append(-1.0 * M)
        return Uc[np.argmax(M_list)]