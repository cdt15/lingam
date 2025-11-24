"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""

import math
import numpy as np
import warnings

import autograd.numpy as anp
from autograd import grad
import scipy.optimize as sopt
import functools
from sklearn.utils import check_array, resample
from sklearn.linear_model import LinearRegression

from .base import _BaseLiNGAM
from .bootstrap import BootstrapResult


class ABICLiNGAM(_BaseLiNGAM):
    """Implementation of ABIC-LiNGAM Algorithm. [1]_
    Original code: https://github.com/Yoshimitsu-try/ABIC_LiNGAM

    References
    ----------
    .. [1] Y. Morinishi and S. Shimizu. Differentiable causal discovery of
       linear non-Gaussian acyclic models under unmeasured confounding.
       Transactions on Machine Learning Research (TMLR), 2025.
    """

    def __init__(
        self,
        beta=1.0,
        lam=0.05,
        acyc_order=None,
        seed=0,
        max_outer=100,
        tol_h=1e-8,
        min_causal_effect=0.05,
        min_error_covariance=0.05,
        rho_max=1e16,
        inner_start=1,
        inner_growth=1,
        inner_tol=1e-4,
    ):
        """Construct a ABICLiNGAM model.

        Parameters
        ----------
        beta : float, optional (default=1.0)
            Power in residual loss, i.e., ||r||^(2*beta)
        lam : float, optional (default=0.05)
            The weight of the regularization term.
        acyc_order : int or None, optional (default=None)
            Order of the truncated series for acyclicity penalty. If None, defaults to the number of variables.
        min_causal_effect : float, optional (default=0.05)
            Threshold for detecting causal edge.
            Causal edges with absolute values of causal effects less than ``min_causal_effect`` are excluded.
        min_error_covariance : float, optional (default=0.05)
            Threshold for detecting error covariances.
            Error covariances with absolute values less than ``min_error_covariance`` are excluded.
        seed : int, optional (default=0)
            Seed for the random number generator.
        max_outer : int, optional (default=100)
            Maximum number of outer iterations.
        tol_h : float, optional (default=1e-8)
            Tolerance for acyclicity penalty to stop.
        rho_max : float, optional (default=1e16)
            Maximum value for Augmented Lagrangian penalty parameter rho.
        inner_start : int, optional (default=1)
            Initial number of inner refinement steps.
        inner_growth : int, optional (default=1)
            Growth of inner refinement steps per outer iteration.
        inner_tol : float, optional (default=1e-4)
            Tolerance for inner loop convergence.
        """
        # Check parameters
        if beta <= 0.0:
            raise ValueError("beta must be positive.")
        if lam < 0.0:
            raise ValueError("lam must be non-negative.")
        if acyc_order is not None:
            if not isinstance(acyc_order, int):
                raise TypeError("acyc_order must be an integer or None.")
            if acyc_order < 1:
                raise ValueError("acyc_order must be >= 1.")
        if min_causal_effect < 0.0:
            raise ValueError("min_causal_effect must be non-negative.")
        if min_error_covariance < 0.0:
            raise ValueError("min_error_covariance must be non-negative.")
        if max_outer < 1:
            raise ValueError("max_outer must be at least 1.")
        if tol_h <= 0.0:
            raise ValueError("tol_h must be positive.")
        if rho_max <= 0.0:
            raise ValueError("rho_max must be positive.")
        if inner_start < 1:
            raise ValueError("inner_start must be at least 1.")
        if inner_growth < 0:
            raise ValueError("inner_growth must be non-negative.")
        if inner_tol <= 0.0:
            raise ValueError("inner_tol must be positive.")

        self._beta = float(beta)
        self._lam = float(lam)
        self._acyc_order = acyc_order
        self._min_causal_effect = float(min_causal_effect)
        self._min_error_covariance = float(min_error_covariance)
        self._seed = int(seed)
        self._max_outer = int(max_outer)
        self._tol_h = float(tol_h)
        self._rho_max = float(rho_max)
        self._inner_start = int(inner_start)
        self._inner_growth = int(inner_growth)
        self._inner_tol = float(inner_tol)

        super().__init__()

    def fit(self, X):
        """Fit the model to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Observed data matrix.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._X = anp.asarray(check_array(X))
        d = self._X.shape[1]

        # Random number generator
        self._rng = np.random.default_rng(self._seed)

        # Check parameters
        if d < 2:
            raise ValueError("Data must have at least two variables (features).")

        # Initialize parameters
        B = anp.array(self._rng.uniform(-0.5, 0.5, size=(d, d)))
        L = anp.array(self._rng.uniform(-0.05, 0.05, size=(d, d)))
        lower_mask = anp.array(np.tril(np.ones((d, d)), k=-1))
        L = L * lower_mask
        L = L + L.T
        L = L - anp.diag(anp.diag(L))
        D = anp.diag(anp.diag(anp.cov(self._X.T)))

        rho, alpha, h_prev = 1.0, 0.0, np.inf
        inner_cap = self._inner_start
        penalty_fn = self._bow_penalty

        bounds = self._build_bounds()  # no prior knowledge
        objective = functools.partial(self._objective)
        gradient = grad(objective)

        for _ in range(self._max_outer):
            B_new, L_new, D_new = None, None, None
            h_new = None

            while rho < self._rho_max:
                B_new = B.copy()
                L_new = L.copy()
                D_new = D.copy()

                # inner refinement
                for _ in range(inner_cap):
                    B_old, L_old, D_old = B_new, L_new, D_new
                    Z = self._pseudo(B_new, L_new + D_new)

                    theta0 = anp.concatenate([anp.ravel(B_new), anp.ravel(L_new)])
                    res = sopt.minimize(
                        self._objective,
                        theta0,
                        args=(rho, alpha, Z, penalty_fn),
                        method="L-BFGS-B",
                        jac=gradient,
                        bounds=bounds,
                        options={"disp": False},
                    )

                    B_new = anp.reshape(res.x[: d * d], (d, d))
                    L_new = anp.reshape(res.x[d * d :], (d, d))
                    L_new = L_new + L_new.T
                    L_new = L_new - anp.diag(anp.diag(L_new))

                    # refresh diagonal noise from residuals (different expression)
                    diag_vals = [
                        anp.var(self._X[:, j] - anp.dot(self._X, B_new[:, j]))
                        for j in range(d)
                    ]
                    D_new = anp.diag(anp.array(diag_vals))

                    # convergence of inner loop
                    delta = anp.sum(anp.abs(B_old - B_new)) + anp.sum(
                        anp.abs((L_old + D_old) - (L_new + D_new))
                    )
                    if float(delta) < self._inner_tol:
                        break

                h_new = self._acyclicity_penalty(B_new) + penalty_fn(B_new, L_new)

                # penalty schedule
                if float(h_new) < 0.25 * float(h_prev):
                    break
                else:
                    rho *= 10.0

            # AL update
            B, L, D = B_new.copy(), L_new.copy(), D_new.copy()
            h_prev = h_new
            alpha = alpha + rho * h_prev
            inner_cap += self._inner_growth

            if float(h_prev) <= self._tol_h or rho >= self._rho_max:
                break

        self._B = anp.where(anp.abs(B) < self._min_causal_effect, 0.0, B)
        self._omega = anp.where(
            anp.abs(L + D) < self._min_error_covariance, 0.0, (L + D)
        )

        # Merge coefficient matrix and error covariance matrix
        omega_copied = self._omega.copy()
        anp.fill_diagonal(omega_copied, 0.0)
        omega_copied[anp.abs(omega_copied) > 0] = anp.nan
        self._adjacency_matrix = self._B.T.copy()
        self._adjacency_matrix[anp.isnan(omega_copied)] = anp.nan

        # Estimate causal order from coefficient matrix
        self._causal_order = self._causal_order_from_adjacency_matrix(self._B.T)

        return self

    def bootstrap(self, X, n_sampling=100):
        """Bootstrap sampling to assess variability of estimates.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Observed data matrix.
        n_sampling : int, optional (default=100)
            Number of bootstrap samples.

        Returns
        -------
        Bs : array-like, shape (n_sampling, n_features, n_features)
            Bootstrap samples of estimated adjacency matrices.
        Omegas : array-like, shape (n_sampling, n_features, n_features)
            Bootstrap samples of estimated error covariance matrices.
        """
        X = anp.asarray(X)
        d = X.shape[1]

        adjacency_matrices = anp.zeros((n_sampling, d, d))
        Bs = anp.zeros((n_sampling, d, d))
        Omegas = anp.zeros((n_sampling, d, d))
        total_effects = anp.zeros((n_sampling, d, d))
        index = anp.arange(X.shape[0])
        resampled_indices = []

        for i in range(n_sampling):
            resampled_X, resampled_index = resample(X, index)
            resampled_indices.append(resampled_index)
            self.fit(resampled_X)
            adjacency_matrices[i] = self._adjacency_matrix
            Bs[i] = self._B
            Omegas[i] = self._omega

            # Calculate total effects
            for c, from_ in enumerate(self._causal_order):
                for to in self._causal_order[c + 1 :]:
                    if True in np.isnan(self._adjacency_matrix[from_]):
                        total_effects[i, to, from_] = np.nan
                    else:
                        total_effects[i, to, from_] = self.estimate_total_effect(
                            resampled_X, from_, to
                        )

        return ABICBootstrapResult(
            adjacency_matrices,
            Bs,
            Omegas,
            total_effects,
            resampled_indices=resampled_indices,
        )

    def _acyclicity_penalty(self, W, K=None):
        """Smooth acyclicity surrogate written as a truncated series:
            h(W) = sum_{k=1..K} trace((W∘W)^k) / k!
        where ∘ is Hadamard product. K defaults to d.
        This differs in form/implementation from common "M=I+..." variants
        and avoids any custom VJP; autograd handles the gradient.

        Parameters
        ----------
        W : array-like, shape (d, d)
            Directed adjacency matrix.
        K : int or None
            Order of the truncated series. If None, defaults to d.

        Returns
        -------
        penalty : float
            Value of the acyclicity penalty.
        """
        d = W.shape[0]
        if K is None:
            K = self._acyc_order or d
        A = W * W
        Ak = anp.eye(d)
        acc = 0.0
        for k in range(1, K + 1):
            Ak = anp.dot(Ak, A)
            acc = acc + anp.trace(Ak) / float(math.factorial(k))
        return acc

    @staticmethod
    def _bow_penalty(W1, W2):
        """Bow-freeness surrogate in an alternative form:
            || W1 ∘ W2 ||_F^2 / |W1|

        Parameters
        ----------
        W1 : array-like, shape (d, d)
            Directed adjacency matrix.
        W2 : array-like, shape (d, d)
            Bidirected adjacency matrix.

        Returns
        -------
        penalty : float
            Value of the bow-freeness penalty.
        """
        A = W1 * W2
        return anp.sum(A * A) / A.size

    def _objective(self, theta, rho, alpha, Z, penalty_fn):
        """Augmented Lagrangian objective. All pieces are auto-diff compatible.

        Parameters
        ----------
        theta : array-like, shape (d*d + d*d,)
            Concatenated parameter vector [vec(B), vec(L)], where L is strictly lower
            triangular part to be mirrored to form a symmetric matrix.
        rho : float
            The weight of the penalty term
        alpha : float
            The Lagrange multiplier
        Z : list of array-like, shape (n, d)
            Pseudo-variables for bidirected part.
        penalty_fn : callable
            Structural penalty function on (B, L_sym).

        Returns
        -------
        obj : float
            Value of the objective function.
        """

        n, d = self._X.shape

        # unpack and enforce symmetry for the bidirected part
        B = anp.reshape(theta[: d * d], (d, d))
        L = anp.reshape(theta[d * d :], (d, d))
        L = L + L.T
        L = L - anp.diag(anp.diag(L))  # zero diagonal

        # LS(theta) term (with generalized power 2*beta)
        LS = 0.0
        for j in range(d):
            r = self._X[:, j] - anp.dot(self._X, B[:, j]) - anp.dot(Z[j], L[:, j])
            LS = LS + 0.5 / n * (anp.linalg.norm(r) ** (2 * self._beta))

        # structural constraints
        h = self._acyclicity_penalty(B) + penalty_fn(B, L)
        aug = 0.5 * rho * (h**2) + alpha * h

        # smooth L0-ish (tanh-like) regularization on theta
        s = anp.log(n) * anp.abs(theta)
        t = (anp.exp(s) - 1) / (anp.exp(s) + 1)
        return LS + aug + self._lam * anp.sum(t)

    def _build_bounds(self, levels=None, exogenous=(), w_range=4.0):
        """Create L-BFGS-B bounds for theta

        Parameters
        ----------
        levels : list[list[index]] or None
            Prior knowledge about variable ordering. Each sublist represents a level,
            where variables in earlier levels cannot have incoming edges from later levels.
        exogenous : array-like, shape (index, ...)
            Indices of exogenous variables (no incoming edges).
        w_range : float
            Range for weights (default: 4.0). Bounds are set to [-w_range, w_range].

        Returns
        -------
        bounds : list[tuple[float, float]]
            Bounds for each parameter in theta = [vec(B), vec(L)].
        """
        d = self._X.shape[1]

        if levels is None:
            levels = [[i for i in range(d)]]

        tier = {v: t for t, group in enumerate(levels) for v in group}
        exo = set(exogenous)

        # Directed bounds B: start wide, zero diag, then forbid backward edges
        B_lo = -w_range * np.ones((d, d))
        B_hi = +w_range * np.ones((d, d))
        np.fill_diagonal(B_lo, 0.0)
        np.fill_diagonal(B_hi, 0.0)
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                if tier[i] > tier[j]:
                    B_lo[i, j] = 0.0
                    B_hi[i, j] = 0.0

        # Bidirected bounds L (we optimize lower-tri; upper/diag fixed zero)
        L_lo = -w_range * np.ones((d, d))
        L_hi = +w_range * np.ones((d, d))
        for i in range(d):
            for j in range(d):
                if i <= j or (i in exo) or (j in exo):
                    L_lo[i, j] = 0.0
                    L_hi[i, j] = 0.0

        bounds_B = np.c_[B_lo.reshape(-1), B_hi.reshape(-1)].tolist()
        bounds_L = np.c_[L_lo.reshape(-1), L_hi.reshape(-1)].tolist()
        return bounds_B + bounds_L

    def _pseudo(self, B, omega):
        """
        Build pseudo-variables Z using solves instead of explicit inversion.

        Parameters
        ----------
        B : array-like, shape (d, d)
            Current estimate of directed adjacency matrix.
        omega : array-like, shape (d, d)
            Current estimate of error covariance matrix.

        Returns
        -------
        Z : list of array-like, shape (n, d)
            Returns a list Z such that Z[j] has a zero column at j (shape (n, d)).
        """
        d = B.shape[0]
        eps = self._X - anp.dot(self._X, B)
        Z = [None] * d
        for j in range(d):
            idx = [k for k in range(d) if k != j]
            omega_ = omega[anp.ix_(idx, idx)]
            Zij = anp.linalg.solve(omega_, eps[:, idx].T).T  # (n, d-1)
            Zj = anp.insert(Zij, j, 0.0, axis=1)  # (n, d)
            Z[j] = Zj
        return Z

    def _causal_order_from_adjacency_matrix(self, matrix, threshold=1e-8):
        """Estimate causal order from adjacency matrix using Kahn's algorithm."""
        d = matrix.shape[0]
        # 有向グラフの隣接リストと入次数を作成
        adj = {i: set() for i in range(d)}
        indegree = [0] * d
        for i in range(d):
            for j in range(d):
                if i != j and abs(matrix[i, j]) > threshold:
                    adj[i].add(j)
                    indegree[j] += 1

        # 入次数0のノードから順に並べる
        order = []
        queue = [i for i in range(d) if indegree[i] == 0]
        while queue:
            n = queue.pop(0)
            order.append(n)
            for m in adj[n]:
                indegree[m] -= 1
                if indegree[m] == 0:
                    queue.append(m)

        if len(order) != d:
            raise ValueError("The adjacency matrix contains a cycle (not a DAG).")

        return order[::-1]

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
            Estimated total effect.
        """
        # Check parameters
        X = check_array(X)

        # Check confounders
        if True in np.isnan(self._adjacency_matrix[from_index]):
            warnings.warn(
                f"The estimated causal effect may be incorrect because "
                f"the source variable (from_index={from_index}) is influenced by confounders."
            )
            return np.nan

        # from_index + parents indices
        parents = np.where(np.abs(self._adjacency_matrix[from_index]) > 0)[0]
        predictors = [from_index]
        predictors.extend(parents)

        # Estimate total effect
        lr = LinearRegression()
        lr.fit(X[:, predictors], X[:, to_index])

        return lr.coef_[0]

    @property
    def coefficient_matrix_(self):
        """Estimated coefficient matrix.

        Returns
        -------
        coefficient_matrix_ : array-like, shape (n_features, n_features)
            The coefficient matrix B of fitted model, where
            n_features is the number of features.
        """
        return self._B

    @property
    def error_covariance_matrix_(self):
        """Estimated error covariance matrix.

        Returns
        -------
        error_covariance_matrix_ : array-like, shape (n_features, n_features)
            The error covariance matrix Omega of fitted model, where
            n_features is the number of features.
        """
        return self._omega


class ABICBootstrapResult(BootstrapResult):
    """The result of bootstrapping for Time series algorithm."""

    def __init__(
        self,
        adjacency_matrices,
        coefficient_matrices,
        error_covariance_matrices,
        total_effects,
        resampled_indices=None,
    ):
        """Construct a BootstrapResult.

        Parameters
        ----------
        adjacency_matrices : array-like, shape (n_sampling)
            The adjacency matrix list by bootstrapping.
        coefficient_matrices : array-like, shape (n_sampling)
            The coefficient matrix list by bootstrapping.
        error_covariance_matrices : array-like, shape (n_sampling)
            The error covariance matrix list by bootstrapping.
        total_effects : array-like, shape (n_sampling)
            The total effects list by bootstrapping.
        resampled_indices : list of array-like, shape (n_sampling), optional (default=None)
            The list of resampled indices used in bootstrapping.
        """
        super().__init__(
            adjacency_matrices, total_effects, resampled_indices=resampled_indices
        )
        self._coefficient_matrices = coefficient_matrices
        self._error_covariance_matrices = error_covariance_matrices

    @property
    def coefficient_matrices_(self):
        """The coefficient matrix list by bootstrapping.

        Returns
        -------
        coefficient_matrices_ : array-like, shape (n_sampling)
            The coefficient matrix list, where ``n_sampling`` is
            the number of bootstrap sampling.
        """
        return self._coefficient_matrices

    @property
    def error_covariance_matrices_(self):
        """The error covariance matrix list by bootstrapping.

        Returns
        -------
        error_covariance_matrices_ : array-like, shape (n_sampling)
            The error covariance matrix list, where ``n_sampling`` is
            the number of bootstrap sampling.
        """
        return self._error_covariance_matrices
