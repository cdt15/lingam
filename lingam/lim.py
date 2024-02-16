"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""

from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                  PoissonRegressor)
from sklearn.metrics import log_loss

import lingam.utils as ut


class LiM:
    """Implementation of LiM Algorithm [1]_

    References
    ----------
    .. [1] Zeng Y, Shimizu S, Matsui H, et al. Causal discovery for
       linear mixed data[C]//Conference on Causal Learning and Rea-
       soning. PMLR, 2022: 994-1009.
    """

    def __init__(
        self,
        lambda1=0.1,
        loss_type="mixed",
        max_iter=150,
        h_tol=1e-8,
        rho_max=1e16,
        w_threshold=0.1,
    ):
        """Construct a LiM model.

        Parameters
        ----------
        lambda1 : float, optional (default=0.1)
             L1 penalty parameter.
        loss_type : str, (default='mixed')
             Type of distribution of the noise.
        max_iter : int, (default=150)
             Maximum number of dual ascent steps.
        h_tol : float, (default=1e-8)
             Tolerance parameter of the acyclicity constraint.
        rho_max : float, (default=1e+16)
             Maximum value of the regularization parameter rho.
        w_threshold : float (default=0.1)
             Drop edge if the weight btw. variables is less than w_threshold.
        """
        self._lambda1 = lambda1
        self._loss_type = loss_type
        self._max_iter = max_iter
        self._h_tol = h_tol
        self._rho_max = rho_max
        self._w_threshold = w_threshold
        self._adjacency_matrix = None

    def fit(self, X, dis_con, only_global=False, is_poisson=False):
        """Fit the model to X with mixed data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of observed variables.
        dis_con: array-like, shape (1, n_features)
            Indicators of discrete or continuous variables, where "1"
            indicates a continuous variable, while "0" a discrete
            variable.
        only_global: boolean, optional (default=False)
            If True, then the method will only perform the global optimization
            to estimate the causal structure, without the local search phase.
        is_poisson: boolean, optional (default=False)
            If True, then the method will use poisson regression model to compute
            the log-likelihood in the local search phase.

        Returns
        -------
        self : object
            Returns the instance of self.
        """

        W_min_lss = self._estimate_LiM(X, dis_con, only_global, is_poisson)
        self._adjacency_matrix = W_min_lss
        return self._adjacency_matrix

    def _estimate_LiM(self, X, dis_con, only_global, is_poisson):
        """Estimate the adjacency matrix btw. mixed variables"""

        def _loss(W):
            """Evaluate value and gradient of loss."""
            M = X @ W
            if self._loss_type == "logistic":
                loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
                G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
            elif self._loss_type == "poisson":
                loss = -np.sum(X.T @ M) + np.exp(M).sum()
                for j in range(X.shape[0]):
                    for k in range(X.shape[1]):
                        loss += np.log(_factorial(X[j][k]))
                loss = 1.0 / X.shape[0] * loss
                G_loss = 1.0 / X.shape[0] * (-X.T @ X + X.T @ np.exp(M))
            elif self._loss_type == "laplace":
                R = X - M
                loss = -1.0 / X.shape[0] * np.sum(-np.log(np.cosh(R)))
                G_loss = 1.0 / X.shape[0] * X.T @ np.tanh(R)
            elif self._loss_type == "mixed":
                R = X - M
                a1, a2 = 1, 1
                loss_dis = (
                    (np.logaddexp(0, M) - X * M) * np.absolute(dis_con - 1)
                ).sum()
                loss_con = np.sum(-np.log(np.cosh(R)) * dis_con)
                loss = -1.0 / X.shape[0] * (a1 * loss_dis + a2 * loss_con)
                W_dis = np.zeros([d, d])
                W_con = np.zeros([d, d])
                for ii in np.where(dis_con[0, :] == 0):
                    W_dis[[ii], :] = 1
                    W_dis[:, [ii]] = 1
                for jj in np.where(dis_con[0, :] == 1):
                    W_con[[jj], :] = 1
                    W_con[:, [jj]] = 1
                G_dis = X.T @ (sigmoid(M) - X) * W_dis
                G_con = -X.T @ np.tanh(R) * W_con
                G_loss = 1.0 / X.shape[0] * (G_dis + G_con)
            elif self._loss_type == "mixed_dag":
                dag = nx.DiGraph(W)
                lingam_data = np.transpose(X)
                df = pd.DataFrame(X)
                is_continuous = dis_con[0, :].astype(bool)
                is_discrete = np.invert(is_continuous)
                loss = -_bic_scoring(
                    dag, is_discrete, is_poisson, df, lingam_data
                )  # lingam_data:dims*samples
                R = X - M
                G_loss = 1.0 / X.shape[0] * (X.T @ (sigmoid(M) - X) - X.T @ np.tanh(R))
            else:
                raise ValueError("unknown loss type")
            return loss, G_loss

        def _bic_scoring(dag: nx.DiGraph, is_discrete, is_poisson, df, lingam_data):
            """Evaluate value of loss given the DAG."""
            sample_size = df.shape[0]
            K = dag.number_of_edges() + dag.number_of_nodes()
            # K = dag.number_of_edges() + np.sum(is_discrete)
            penalty = np.log(sample_size) / 2 * K
            total_score = 0.0 - penalty

            for i in dag.nodes:
                parents_i = [j for j in dag.predecessors(i)]

                # 離散 logistic
                if is_discrete[i] and is_poisson == False:
                    if not parents_i:
                        # Bernoulli binary variable, likelihood using Bernoulli model with MLE parameters
                        frequency_table = df[i].value_counts()
                        likekihood_bernoulli = 0.0
                        for count_k in frequency_table:
                            likekihood_bernoulli += count_k * (
                                np.log(count_k) - np.log(frequency_table.sum())
                            )
                        total_score += likekihood_bernoulli

                    elif parents_i:
                        # Logistic binary variable, likelihood using logistic regression model.
                        X = df[parents_i]
                        y = df[i]
                        logistic = LogisticRegression(
                            solver="lbfgs"
                        )  # or solver='liblinear'
                        logistic.fit(X, y)
                        predict_prob = logistic.predict_proba(X)

                        # Negative cross-entropy loss a.k.a log-likelihood
                        likekihood_logistic = -log_loss(
                            y_true=y, y_pred=predict_prob, normalize=False
                        )
                        total_score += likekihood_logistic
                    pass

                # 離散 poisson
                elif is_discrete[i] and is_poisson == True:
                    if not parents_i:
                        # Bernoulli count variable, likelihood using Bernoulli model with MLE parameters
                        frequency_table = df[i].value_counts()
                        likekihood_bernoulli = 0.0
                        for count_k in frequency_table:
                            likekihood_bernoulli += count_k * (
                                np.log(count_k) - np.log(frequency_table.sum())
                            )
                        total_score += likekihood_bernoulli

                    elif parents_i:
                        # Count variable, likelihood using poisson regression model.
                        for iii in range(len(parents_i)):
                            X = lingam_data[parents_i[iii]]
                            X = X.reshape(-1, 1)
                            y = lingam_data[i]
                            poisson = PoissonRegressor()
                            poisson.fit(X, y)
                            beta = poisson.coef_
                            # compute likelihood
                            likekihood_poisson = -np.sum(
                                y * lingam_data[parents_i[iii]] * beta
                            ) + np.sum(np.exp(lingam_data[parents_i[iii]] * beta))
                            for j in range(len(y)):
                                likekihood_poisson += np.log(_factorial(y[j]))
                        total_score += likekihood_poisson
                        # # or we can compute likelihood via nn
                        # pnllloss = nn.PoissonNLLLoss()
                        # log_input = np.log(lingam_data[parents_i] * beta)
                        # target = y # torch.randn(5, 2)
                        # output = pnllloss(log_input, target)
                        # likekihood_poisson = output.item()
                        # total_score += likekihood_poisson

                # 連続
                elif not is_discrete[i]:
                    b_i = np.zeros(dag.number_of_nodes())

                    if parents_i:  # estimate b_i
                        X = df[parents_i]
                        y = df[i]
                        lr = LinearRegression(fit_intercept=True)  # because zero means
                        lr.fit(X, y)
                        for index, j in enumerate(parents_i):
                            b_i[j] = lr.coef_[index]

                        bi_0 = lr.intercept_
                    else:
                        bi_0 = np.mean(df[i])

                    likekihood_lingam = ut.likelihood_i(lingam_data, i, b_i, bi_0)
                    total_score += likekihood_lingam
                    pass
            return total_score

        def _h(W):
            """Evaluate value and gradient of acyclicity constraint."""
            # E = slin.expm(W * W)  # (Zheng et al. 2018)
            # h = np.trace(E) - d
            # A different formulation, slightly faster at the cost of numerical stability
            M = np.eye(d) + W * W / d  # (Yu et al. 2019)
            G_h = np.linalg.matrix_power(M, d - 1)
            h = (G_h.T * M).sum() - d
            return h, G_h

        def _adj(w):
            """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
            return (w[: d * d] - w[d * d :]).reshape([d, d])

        def _func(w):
            """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
            W = _adj(w)
            loss, G_loss = _loss(W)
            h, G_h = _h(W)
            obj = loss + 0.5 * rho * h * h + alpha * h + self._lambda1 * w.sum()
            # G_smooth = G_loss + (rho * h + alpha) * G_h # Zheng Xun Dag 2018
            G_smooth = G_loss + (rho * h + alpha) * G_h.T * W * 2  # 2019
            g_obj = np.concatenate(
                (G_smooth + self._lambda1, -G_smooth + self._lambda1), axis=None
            )
            return obj, g_obj

        def _factorial(y):
            if not isinstance(y, int):
                y = round(y)
            if y == 0 or y == 1 or y < 0:
                return 1
            else:
                return y * _factorial(y - 1)

        n, d = X.shape
        w_est, rho, alpha, h = (
            np.random.random(2 * d * d),
            1.0,
            0.0,
            np.inf,
        )  # double w_est into (w_pos, w_neg)
        bnds = [
            (0, 0) if i == j else (0, None)
            for _ in range(2)
            for i in range(d)
            for j in range(d)
        ]

        # if self._loss_type == 'l2':
        #     X = X - np.mean(X, axis=0, keepdims=True)
        for _ in range(self._max_iter):
            w_new, h_new = None, None
            while rho < self._rho_max:
                sol = sopt.minimize(
                    _func, w_est, method="L-BFGS-B", jac=True, bounds=bnds
                )
                # print('--- One iteration passed.....', sol.fun)
                w_new = sol.x
                h_new, _ = _h(_adj(w_new))
                if h_new >= 0.25 * h:
                    rho *= 10
                else:
                    break
            w_est, h = w_new, h_new
            alpha += rho * h
            # print('------- rho  is:', rho)
            if h <= self._h_tol and h != 0:
                break
            if rho >= self._rho_max * 1e-6 and h > 1e05:  # avoid the full graph
                w_est = np.random.random(2 * d * d)
                rho = 1.0
            elif rho >= self._rho_max:
                break
            if np.sum(np.absolute(_adj(w_est))) < 0.09:  # avoid the zero matrix
                w_est = np.random.random(2 * d * d)
        W_est = _adj(w_est)
        W_est[np.abs(W_est) < self._w_threshold] = 0
        print("W_est (without the 2nd phase) is: \n", W_est)

        if not only_global:
            self._loss_type = "mixed_dag"
            aa, aaa = _loss(W_est)  # loss
            I = np.where(W_est != 0)
            # check directions
            W_min_lss = np.copy(W_est)
            candi_setting = list(
                product(range(2), repeat=len(I[0]))
            )  # 1:reverse the direction； 0:unchanged
            for candi_setting_i in range(1, len(candi_setting)):
                W_tmp = np.copy(W_est)
                for iii in range(len(I[0])):  # transform to W_tmp
                    if candi_setting[candi_setting_i][iii] == 1:
                        W_tmp[I[0][iii], I[1][iii]] = 0
                        W_tmp[I[1][iii], I[0][iii]] = 1
                lss, lss_G = _loss(W_tmp)
                if lss < aa and _h(W_tmp)[0] < self._h_tol:
                    W_min_lss = np.copy(W_tmp)
                    aa = lss

            # prune process
            if d > 2 and len(I[0]) > (d - 1):  # > min_edge
                W0 = np.copy(W_min_lss)
                I_delete = np.where(W_min_lss != 0)
                for delete_i in range(len(I_delete[0])):
                    W_tmp = np.copy(W0)
                    W_tmp[I_delete[0][delete_i], I_delete[1][delete_i]] = 0
                    lss, lss_G = _loss(W_tmp)
                    if lss < aa and _h(W_tmp)[0] < self._h_tol:
                        W_min_lss = np.copy(W_tmp)
                        aa = lss
                #
                if not np.all(
                    W_est.astype(bool) == W_min_lss.astype(bool)
                ):  # if they are different
                    W0 = np.copy(W_est)
                    for delete_i in range(len(I[0])):
                        W_tmp = np.copy(W0)
                        W_tmp[I[0][delete_i], I[1][delete_i]] = 0
                        lss, lss_G = _loss(W_tmp)
                        if lss < aa and _h(W_tmp)[0] < self._h_tol:
                            W_min_lss = np.copy(W_tmp)
                            aa = lss
            # add process
            if d > 2 and len(I[0]) < (d * (d - 1) / 2):
                W0 = np.copy(W_min_lss)
                W_edges = W0 + W0.T + np.eye(d)
                I_add = np.where(W_edges == 0)  # add undirected edges' indices
                for add_i in range(len(I_add[0])):
                    W_tmp = np.copy(W0)
                    W_tmp[I_add[0][add_i], I_add[1][add_i]] = 1
                    lss, lss_G = _loss(W_tmp)
                    if lss < aa and _h(W_tmp)[0] < self._h_tol:
                        W_min_lss = np.copy(W_tmp)
                        aa = lss
                #
                if not np.all(
                    W_est.astype(bool) == W_min_lss.astype(bool)
                ):  # if they are different
                    W0 = np.copy(W_est)
                    W_edges = W0 + W0.T + np.eye(d)
                    I_add = np.where(W_edges == 0)  # add undirected edges' indices
                    for add_i in range(len(I_add[0])):
                        W_tmp = np.copy(W0)
                        W_tmp[I_add[0][add_i], I_add[1][add_i]] = 1
                        lss, lss_G = _loss(W_tmp)
                        if lss < aa and _h(W_tmp)[0] < self._h_tol:
                            W_min_lss = np.copy(W_tmp)
                            aa = lss
            # print('W_min_lss is:\n', W_min_lss)
            # print('W_true is:\n', W_true)

        else:
            W_min_lss = np.copy(W_est)

        return W_min_lss

    @property
    def adjacency_matrix_(self):
        """Estimated adjacency matrix between mixed variables.

        Returns
        -------
        adjacency_matrix_ : array-like, shape (n_features, n_features)
            The adjacency matrix of variables, where
            ``n_features`` is the number of observed variables.
        """
        return self._adjacency_matrix
