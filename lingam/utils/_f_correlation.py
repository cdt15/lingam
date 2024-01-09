"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from sklearn.utils import check_array


def f_correlation(x, y):
    """Implementation of F-correlation [2]_

    References
    ----------
    .. [2] Bach, Francis R., and Michael I. Jordan. "Kernel independent component analysis."
        Journal of machine learning research 3.Jul (2002): 1-48.

    Parameters
    ----------
    x, y : array-like, shape (n_samples)
        Data, where ``n_samples`` is the number of samples.

    Returns
    -------
    float
        The valus of F-correlation.
    """

    x_ = check_array(x, ensure_2d=False)
    y_ = check_array(y, ensure_2d=False)
    n = x_.shape[0]

    if y_.shape[0] != n:
        raise ValueError("x and y must be the same size.")

    # Standardize
    x_ = (x_ - x_.mean()) / x_.std()
    y_ = (y_ - y_.mean()) / y_.std()

    X = [x_, y_]
    m = len(X)

    if n > 1000:
        kappa, sigma = [2e-3, 0.5]
    else:
        kappa, sigma = [2e-2, 1.0]

    Rs = []
    Us = []
    sizes = []
    for i in range(m):
        # Incomplete Cholesky decomposition
        G, P = _incomplete_cholesky(X[i], sigma, n * kappa * 1e-2)
        G = G[np.argsort(P), :]
        G = G - np.tile(np.mean(G, axis=0), (len(G), 1))

        # Singular value decomposition
        R, U = _svd(G, n, kappa)
        Rs.append(R)
        Us.append(U)
        sizes.append(R.shape[0])

    # make R_kappa
    R_kappa = np.eye(sum(sizes))
    st = np.cumsum(np.hstack([0, sizes]))
    st = st[:m]
    for i in range(1, m):
        for j in range(0, i):
            RUUR = np.diag(Rs[i]) @ (Us[i].T @ Us[j]) @ np.diag(Rs[j])
            R_kappa[st[i] : st[i] + sizes[i], st[j] : st[j] + sizes[j]] = RUUR
            R_kappa[st[j] : st[j] + sizes[j], st[i] : st[i] + sizes[i]] = RUUR.T

    # smallest eigenvalue of R_kappa
    w, _ = eigsh(R_kappa, 1, which="SM")

    return 1 - w[0]


def _squared_dist(x, y):
    """Squared euclidean distance matrix."""
    xx = x * x
    yy = y * y
    xy = x.reshape(-1, 1) * y
    xx = np.tile(xx.reshape(-1, 1), (1, len(yy)))
    yy = np.tile(yy, (len(xx), 1))
    return np.abs(xx + yy - 2 * xy)


def _incomplete_cholesky(x, sigma, tol):
    """Incomplete cholesky decomposition of the gram matrix with the gaussian kernel."""
    n = x.shape[0]
    i = 0
    Gjj = np.ones((n))
    G = np.empty((n, 0))
    P = np.array([i for i in range(n)])

    while np.sum(Gjj[i:n]) > tol:
        G = np.append(G, np.zeros((n, 1)), axis=1)
        if i > 0:
            jast = np.argmax(Gjj[i:n])
            jast = jast + i
            P[[i, jast]] = P[[jast, i]]
            G[[i, jast], :i] = G[[jast, i], :i]
        else:
            jast = 0

        G[i, i] = Gjj[jast]
        G[i, i] = np.sqrt(G[i, i])

        if i < n:
            dist = _squared_dist(x[P[(i + 1) : n]], x[[P[i]]])
            K = np.exp(-0.5 / sigma**2 * dist)
            if i > 0:
                sigG = G[(i + 1) : n, 0:i] @ G[i, 0:i].reshape(-1, 1)
                G[(i + 1) : n, [i]] = 1 / G[i, i] * (K - sigG)
            else:
                G[(i + 1) : n, [i]] = 1 / G[i, i] * K

        if i < n:
            Gjj[(i + 1) : n] = np.ones((n - (i + 1))) - np.sum(
                G[(i + 1) : n, 0 : i + 1] ** 2, axis=1
            )

        i = i + 1

    return G, P


def _svd(G, n, kappa):
    """Singular value decomposition."""
    eta = kappa * 1e-2
    D, A = eigh(G.T @ G)
    indexes = np.where(D >= n * eta)[0]
    order = np.argsort(D[indexes])
    order = order[::-1]
    indexes = indexes[order[0 : len(indexes)]]
    D = D[indexes]
    U = G @ (A[:, indexes] @ np.diag(np.sqrt(1.0 / (D))))
    R = D
    for j in range(len(D)):
        # regularized
        R[j] = D[j] / (n * kappa / 2 + D[j])
    return R, U
