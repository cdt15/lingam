"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""

import numpy as np
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
# from factor_analyzer import (ConfirmatoryFactorAnalyzer, ModelSpecificationParser)
from psy import cfa
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd


class LiNA:
    """Implementation of LiNA Algorithm [1]_

    References
    ----------
    .. [1] Y. Zeng, S. Shimizu, R. Cai, F. Xie, M. Yamamoto and Z. Hao.
       Causal Discovery with Multi-Domain LiNGAM for Latent Factors.
       In Proc. of the Thirtieth International Joint Conference on
       Artificial Intelligence (IJCAI-21), 2021: 2097--2103.
    """

    def __init__(
            self,
            w_threshold=0.3,
            lambda1=0.1,
            lambda2=0.1,
            loss_type='laplace',
            max_iter=100,
            h_tol=1e-8,
            rho_max=1e+16
    ):
        """Construct a LiNA model.

        Parameters
        ----------
         w_threshold : float (default=0.3)
             Drop edge if the weight btw. latent factors is less than w_threshold.
         lambda1 : float, optional (default=0.1)
             L1 penalty parameter.
         lambda2 : float, (default=0.1)
             L2 penalty parameter.
         loss_type : str, (default='laplace')
             Type of distribution of the noise.
         max_iter : int, (default=100)
             Maximum number of dual ascent steps.
         h_tol : float, (default=1e-8)
             Tolerance parameter of the acyclicity constraint.
         rho_max : float, (default=1e+16)
             Maximum value of the regularization parameter rho.
        """

        self._w_threshold = w_threshold
        self._lambda1 = lambda1
        self._lambda2 = lambda2
        self._loss_type = loss_type
        self._max_iter = max_iter
        self._h_tol = h_tol
        self._rho_max = rho_max
        self._adjacency_matrix = None
        self._measurement_matrix = None

    def fit(self, X, G_sign, scale):
        """Fit the model to X with measurement structure and latent factors' scales.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of measurement features.
        G_sign: array-like, shape (n_features, n_features_latent)
            Measurement structure matrix, where ``n_features_latent`` is
            the number of  latent factors and ``n_features`` is the
            number of measurement features.
        scale: array-like, shape (1, n_features_latent)
            Scales of the latent factors.

        Returns
        -------
        self : object
            Returns the instance of self.
        """

        n_samples = X.shape[0]
        n_features_latent = G_sign.shape[1]

        lam, phi, var_e = cfa(X, G_sign)
        G_est_nor = lam

        f_est_nor = np.linalg.inv(G_est_nor.T @ G_est_nor) @ G_est_nor.T @ (X.T)
        f_est_nor = f_est_nor.T
        f_est = np.zeros([n_samples, n_features_latent])
        for j in range(n_features_latent):
            f_est[:, j] = f_est_nor[:, j] * scale[0, j]

        # W_est = _estimate_LiNA(f_est, self._w_threshold, self._lambda1, self._loss_type)  # input f: n*d
        f = f_est
        W_est = self._estimate_LiNA(f)
        self._adjacency_matrix = W_est
        self._measurement_matrix = G_est_nor
        return self._adjacency_matrix

    def _estimate_LiNA(self, f):
        """Estimate the adjacency matrix btw. estimated latent factors for single-domain data"""

        def _loss(W):
            """Evaluate value and gradient of loss."""
            M = f @ W
            if self._loss_type == 'l2':
                R = f - M
                loss = 0.5 / f.shape[0] * (R ** 2).sum()
                D = - 1.0 / f.shape[0] * f.T @ R
            elif self._loss_type == 'logistic':
                loss = 1.0 / f.shape[0] * (np.logaddexp(0, M) - f * M).sum()
                D = 1.0 / f.shape[0] * f.T @ (sigmoid(M) - f)
            elif self._loss_type == 'poisson':
                S = np.exp(M)
                loss = 1.0 / f.shape[0] * (S - f * M).sum()
                D = 1.0 / f.shape[0] * f.T @ (S - f)
            elif self._loss_type == 'laplace':
                R = f - M
                loss = - np.sum(-np.log(np.cosh(R)))
                D = - f.T @ np.tanh(R)
            else:
                raise ValueError('unknown loss type')
            return loss, D

        def _h(W):
            """Evaluate value and gradient of acyclicity constraint."""
            #     E = slin.expm(W * W)  # (Zheng et al. 2018)
            #     h = np.trace(E) - d
            M = np.eye(d) + W * W / d  # (Yu et al. 2019)
            E = np.linalg.matrix_power(M, d - 1)
            h = (E.T * M).sum() - d
            return h, E

        def _adj(w):
            """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
            return (w[:d * d] - w[d * d:]).reshape([d, d])

        def _func(w):
            """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
            W = _adj(w)
            loss, D = _loss(W)
            h, E = _h(W)
            obj = loss + 0.5 * rho * h * h + alpha * h + self._lambda1 * w.sum()
            # obj = loss + 0.5 * rho * h * h + alpha * h + self._lambda1 * w.sum()
            # + self._lambda2*(np.linalg.norm(W, ord = 'fro')**2) # add l2 penalty constraint
            G = D + (rho * h + alpha) * E.T * W * 2
            # G = D + (rho * h + alpha) * E.T * W * 2 + 2 * self._lambda2 * W #gradient of loss + h + penalty
            grad_cat = np.concatenate((G + self._lambda1, - G + self._lambda1), axis=None)
            return obj, grad_cat

        n, d = f.shape
        w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
        bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
        for _ in range(self._max_iter):
            w_new, h_new = None, None
            while rho < self._rho_max:
                sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
                w_new = sol.x
                h_new, _ = _h(_adj(w_new))
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break
            w_est, h = w_new, h_new
            alpha += rho * h
            if h <= self._h_tol or rho >= self._rho_max:
                break
        W_est = _adj(w_est)
        W_est[np.abs(W_est) < self._w_threshold] = 0

        return W_est

    @property
    def adjacency_matrix_(self):
        """Estimated adjacency matrix between latent factors.

        Returns
        -------
        adjacency_matrix_ : array-like, shape (n_features_latent, n_features_latent)
            The adjacency matrix of latent factors, where
            ``n_features_latent`` is the number of latent
            factors.
        """
        return self._adjacency_matrix

    @property
    def measurement_matrix_(self):
        """Estimated measurement matrix between measurement variables and
        latent factors.

        Returns
        -------
        measurement_matrix_ : array-like, shape (n_features, n_features_latent)
            The measurement matrix between measurement variables and
            latent factors, where ``n_features_latent`` is the
            number of latent factors and ``n_features`` is the
            number of measurement variables.
        """
        return self._measurement_matrix


class MDLiNA:
    """Implementation of MD-LiNA Algorithm [2]_

    References
    ----------
    .. [2] Y. Zeng, S. Shimizu, R. Cai, F. Xie, M. Yamamoto and Z. Hao.
       Causal Discovery with Multi-Domain LiNGAM for Latent Factors.
       In Proc. of the Thirtieth International Joint Conference on
       Artificial Intelligence (IJCAI-21), 2021: 2097--2103.
    """

    def __init__(
            self,
            w_threshold=0.3,
            lambda1=0.1,
            lambda2=0.1,
            loss_type='laplace',
            max_iter=100,
            h_tol=1e-8,
            rho_max=1e+16,
            no_of_domain=2,
            no_of_latent_1domain=3,
    ):
        """Construct an MD-LiNA model.

        Parameters
        ----------
         w_threshold : float (default=0.3)
             Drop edge if the weight btw. latent factors is less than w_threshold.
         lambda1 : float, optional (default=0.1)
             L1 penalty parameter.
         lambda2 : float, (default=0.1)
             L2 penalty parameter.
         loss_type : str, (default='laplace')
             Type of distribution of the noise.
         max_iter : int, (default=100)
             Maximum number of dual ascent steps.
         h_tol : float, (default=1e-8)
             Tolerance parameter of the acyclicity constraint.
         rho_max : float, (default=1e+16)
             Maximum value of the regularization parameter rho.
         no_of_domain : int, (default=2)
             Number of domains.
         no_of_latent_1domain : float, (default=3)
             Number of latent factors in a domain.

        """

        self._w_threshold = w_threshold
        self._lambda1 = lambda1
        self._lambda2 = lambda2
        self._loss_type = loss_type
        self._max_iter = max_iter
        self._h_tol = h_tol
        self._rho_max = rho_max
        self._no_of_domain = no_of_domain
        self._no_of_latent_1domain = no_of_latent_1domain
        self._adjacency_matrix = None
        self._measurement_matrix = None

    def fit(self, X, G_sign, scale):
        """Fit the model to X with measurement structure and latent factors' scales.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            of all domains and ``n_features`` is the number of features
            of all domains.
        G_sign: array-like, shape (n_features, n_features_latent)
            Measurement structure matrix, where ``n_features_latent`` is
            the number of latent factors of all domains and ``n_features``
            is the number of measurement variables of all domains.
        scale: array-like, shape (1, n_features_latent)
            Scales of the latent factors.

        Returns
        -------
        self : object
            Returns the instance of self.
        """

        # # PHASE I: use CFA to estimate \bar{G}
        lam, phi, var_e = cfa(X, G_sign)
        G_est = lam

        f_est = np.linalg.inv(G_est.T @ G_est) @ G_est.T @ (X.T)
        f_est = f_est.T
        # F = f_est

        F = np.zeros([f_est.shape[0], f_est.shape[1]])
        for j in range(f_est.shape[1]):
            if j < self._no_of_latent_1domain:
                F[:, j] = f_est[:, j] * scale[0, j]
            else:
                F[:, j] = f_est[:, j] * scale[1, j]

        # optimize together
        W0 = np.zeros([self._no_of_latent_1domain, self._no_of_latent_1domain])
        H0 = np.zeros([self._no_of_latent_1domain * self._no_of_domain, self._no_of_latent_1domain])
        times = 0
        min_like = np.inf
        Likelihood_all = np.zeros([1, 10000])

        while 1:
            # update H
            H_est = self._estimate_MDLiNA_H(F, W0)
            F_final2 = np.linalg.inv(H_est.T @ H_est) @ H_est.T @ F.T

            # update B
            W_est = self._estimate_MDLiNA_B(F_final2.T)
            times = times + 1

            print('W_est is:\n', W_est)
            print('W0 is:\n', W0)

            # check if convergent?
            inv_HH = np.linalg.inv(H_est.T @ H_est) @ H_est.T
            P_H = H_est @ inv_HH @ F.T
            LL = np.sum(np.log(np.cosh(F @ H_est - F @ inv_HH.T @ W_est))) + (np.abs(W_est)).sum() + 10000 * self._conh(
                W_est)
            recon_error = np.trace((F.T - P_H).T @ (F.T - P_H))
            H1 = (np.abs(H_est)).sum()
            likelihood_all = LL + recon_error + H1
            Likelihood_all[0, times] = likelihood_all
            if likelihood_all <= min_like:
                min_like = likelihood_all
            if np.abs(Likelihood_all[0, times] - Likelihood_all[0, times - 1]) < 1e-1:
                break
            elif times >= 30 and np.abs(likelihood_all - min_like) < 1e-1:
                break
            elif times >= 30 and likelihood_all <= min_like:
                break
            elif times >= 50 and np.abs(likelihood_all - np.min(Likelihood_all[0, times - 20:times - 1])) < 1e-1:
                break
            elif times >= 50 and likelihood_all <= np.min(Likelihood_all[0, times - 20:times - 1]):
                break
            else:
                W0 = W_est
                H0 = H_est

        if np.where(H_est != 0)[0].shape[0] == 0:
            H_est = H0
        if np.where(W_est != 0)[0].shape[0] == 0:
            W_est = W0

        # choose new factors: represent which concept.
        tt = 0
        if tt == 0:
            Weight = np.abs(H_est)
            W = W_est
        else:
            Weight = np.abs(H_est)
            W = W0
        BB = np.zeros([self._no_of_latent_1domain, self._no_of_latent_1domain])
        for i in range(self._no_of_latent_1domain):
            for j in range(self._no_of_latent_1domain):
                BB[j, i] = np.abs(Weight[j, i]) + np.abs(Weight[j + self._no_of_latent_1domain, i])
        order = np.zeros([1, self._no_of_latent_1domain])
        if len(np.where(BB == 0)) == 0:
            for i in range(self._no_of_latent_1domain):
                indice = np.where(BB == np.amax(BB))
                order[0, indice[1]] = indice[0]
                BB[indice[0], :] = 0
                BB[:, indice[1]] = 0
        else:
            indice = np.where(BB == 0)
            BB[:, indice[1]] = BB[:, indice[1]] + 0.01
            for i in range(self._no_of_latent_1domain):
                indice = np.where(BB == np.amax(BB))
                if len(indice[0]) > 1:
                    order[0, indice[1][0]] = indice[0][0]
                    BB[indice[0][0], :] = 0
                    BB[:, indice[1][0]] = 0
                else:
                    order[0, indice[1]] = indice[0]
                    BB[indice[0], :] = 0
                    BB[:, indice[1]] = 0

        W_est_new0 = np.zeros([self._no_of_latent_1domain, self._no_of_latent_1domain])
        W_est_new = np.zeros([self._no_of_latent_1domain, self._no_of_latent_1domain])
        for i in range(self._no_of_latent_1domain):
            indice = np.where(order == i)
            W_est_new0[[i], :] = W[indice[1], :]  # row
        for i in range(self._no_of_latent_1domain):
            indice = np.where(order == i)
            W_est_new[:, [i]] = W_est_new0[:, indice[1]]  # column

        self._adjacency_matrix = W_est_new
        self._measurement_matrix = G_est

        return self._adjacency_matrix

    def calc_vif(X):
        # Calculating VIF
        vif = pd.DataFrame()
        X = pd.DataFrame(X)  # convert ndarray to dataframe
        vif["variables"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        return (vif)

    def _estimate_MDLiNA_H(self, f, W):
        """ Optimize H matrix btw. multi-domain latent factors and latent factors of interest

            Returns:
                H_est (np.ndarray): [hq, q] estimated H
            """

        def _loss(H):
            """Evaluate value and gradient of loss."""
            inv_HH = np.linalg.inv(H.T @ H)
            fH = f @ H
            M = fH @ inv_HH @ W
            if self._loss_type == 'l2':
                R = f - M
                loss = 0.5 / f.shape[0] * (R ** 2).sum()
                D = - 1.0 / f.shape[0] * f.T @ R
            elif self._loss_type == 'laplace':
                R = fH @ inv_HH - M
                # standardize
                for j in range(hq):
                    R[:, j] = R[:, j] - np.mean(R[:, j])
                    R[:, j] = R[:, j] / np.std(R[:, j])
                if np.trace(H.T @ H) < 10000:
                    aa = np.trace(H.T @ H)
                    D = - f.T @ np.tanh(R) @ (np.eye(np.size(H, 1)) - W) @ inv_HH \
                        + H @ inv_HH @ inv_HH @ fH.T @ np.tanh(R) @ (np.eye(np.size(H, 1)) - W) \
                        + H @ inv_HH @ inv_HH @ (np.eye(np.size(H, 1)) - W).T @ np.tanh(R).T @ fH \
                        - 2 * f.T @ fH @ inv_HH + 2 * H @ inv_HH @ fH.T @ fH @ inv_HH
                else:
                    aa = 0
                    D = - f.T @ np.tanh(R) @ (np.eye(np.size(H, 1)) - W) @ inv_HH \
                        + H @ inv_HH @ inv_HH @ fH.T @ np.tanh(R) @ (np.eye(np.size(H, 1)) - W) \
                        + H @ inv_HH @ inv_HH @ (np.eye(np.size(H, 1)) - W).T @ np.tanh(R).T @ fH \
                        - 2 * f.T @ fH @ inv_HH + 2 * H @ inv_HH @ fH.T @ fH @ inv_HH + 2 * H
                loss = np.sum(np.log(np.cosh(R))) \
                    + np.trace((f - fH @ inv_HH @ H.T).T @ (f - fH @ inv_HH @ H.T)) \
                    + (np.trace(H.T @ H) - aa)
            else:
                raise ValueError('unknown loss type')
            return loss, D

        def _adj(h):
            """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
            return (h[:hq * q] - h[hq * q:]).reshape([q, hq])

        def _func(h):
            """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
            H = _adj(h)
            loss, D = _loss(H)
            obj = loss + self._lambda1 * h.sum()
            G = D  # gradient of loss
            grad_cat = np.concatenate((G + self._lambda1, - G + self._lambda1), axis=None)
            return obj, grad_cat

        n, q = f.shape
        hq = np.size(W, 0)  # hq is \hat{q}: the number of shared latent factors
        # ut.set_random_seed(0)
        h_est, score_0 = np.random.rand(2 * hq * q), np.inf
        for _ in range(self._max_iter):
            # h_new, score_1 = None, None
            sol = sopt.minimize(_func, h_est, method='BFGS', jac=True)
            h_new = sol.x
            score_1 = sol.fun

            if np.abs(score_0 - score_1) < self._h_tol:
                break
            if score_1 > 0 and score_1 < score_0:
                h_est = h_new
                score_0 = score_1
                # see if singular?
                bb = _adj(h_est)
                bb[np.abs(bb) < self._w_threshold] = 0
                if np.linalg.det(bb.T @ bb) == 0:
                    h_est[0:8] = h_est[0:8] + 0.1
                if np.max(np.abs(h_est)) > 100:
                    h_est = np.random.rand(2 * hq * q)
                    score_0 = np.inf
            else:
                h_est = np.random.rand(2 * hq * q)

        h_est = h_new
        H_est = _adj(h_est)
        H_est[np.abs(H_est) < self._w_threshold] = 0
        self._H_est = H_est
        return self._H_est

    def _estimate_MDLiNA_B(self, f):
        """Optimize the causal effect matrix among shared latent factors

        Returns:
            W_est (np.ndarray): [d, d]
        """

        def _loss(W):
            """Evaluate value and gradient of loss."""
            M = f @ W
            if self._loss_type == 'l2':
                R = f - M
                loss = 0.5 / f.shape[0] * (R ** 2).sum()
                D = - 1.0 / f.shape[0] * f.T @ R
            elif self._loss_type == 'logistic':
                loss = 1.0 / f.shape[0] * (np.logaddexp(0, M) - f * M).sum()
                D = 1.0 / f.shape[0] * f.T @ (sigmoid(M) - f)
            elif self._loss_type == 'poisson':
                S = np.exp(M)
                loss = 1.0 / f.shape[0] * (S - f * M).sum()
                D = 1.0 / f.shape[0] * f.T @ (S - f)
            elif self._loss_type == 'laplace':
                R = f - M
                loss = - np.sum(-np.log(np.cosh(R)))
                D = - f.T @ np.tanh(R)
            else:
                raise ValueError('unknown loss type')
            return loss, D

        def _h(W):
            """Evaluate value and gradient of acyclicity constraint."""
            M = np.eye(d) + W * W / d  # (Yu et al. 2019)
            E = np.linalg.matrix_power(M, d - 1)
            h = (E.T * M).sum() - d
            return h, E

        def _adj(w):
            """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
            return (w[:d * d] - w[d * d:]).reshape([d, d])

        def _func(w):
            """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
            W = _adj(w)
            loss, D = _loss(W)
            h, E = _h(W)
            obj = loss + 0.5 * rho * h * h + alpha * h + self._lambda1 * w.sum()
            G = D + (rho * h + alpha) * E.T * W * 2
            grad_cat = np.concatenate((G + self._lambda1, - G + self._lambda1), axis=None)
            return obj, grad_cat

        n, d = f.shape
        w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
        bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
        for _ in range(self._max_iter):
            w_new, h_new = None, None
            while rho < self._rho_max:
                sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
                w_new = sol.x
                h_new, _ = _h(_adj(w_new))
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break
            w_est, h = w_new, h_new
            alpha += rho * h
            if h <= self._h_tol or rho >= self._rho_max:
                break
        W_est = _adj(w_est)
        W_est[np.abs(W_est) < self._w_threshold] = 0
        self._W_est = W_est
        return self._W_est

    def _conh(self, W):
        """Evaluate value of acyclicity constraint."""
        d = self._no_of_latent_1domain
        M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        E = np.linalg.matrix_power(M, d - 1)
        h = (E.T * M).sum() - d
        return h

    @property
    def adjacency_matrix_(self):
        """Estimated adjacency matrix between latent factors of interest, which
        is shared by all domains.

        Returns
        -------
        adjacency_matrix_ : array-like, shape (n_features_latent_1domain, n_features_latent_1domain)
            The adjacency matrix of latent factors of interest, where
            ``n_features_latent_1domain`` is the number of latent
            factors of interest.
        """
        return self._adjacency_matrix

    @property
    def measurement_matrix_(self):
        """Estimated measurement matrix between measurement variables and
        latent factors from all domains.

        Returns
        -------
        measurement_matrix_ : array-like, shape (n_features, n_features_latent)
            The measurement matrix between measurement variables and
            latent factors, where ``n_features_latent`` is the
            number of latent factors and ``n_features`` is the
            number of measurement variables from all domains.
        """
        return self._measurement_matrix
