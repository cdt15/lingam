import os

import numpy as np
import pandas as pd
from lingam.abic_lingam import ABICLiNGAM
from lingam.utils._mggd import MGGD


def test_fit_success():
    d = 4
    B_true = np.zeros((d, d), dtype=float)
    B_true[0, 1] = 1.0  # x0 -> x1
    B_true[1, 2] = -1.5  # x1 -> x2
    B_true[2, 3] = 1.0  # x2 -> x3

    omega_true = np.array(
        [
            [1.2, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.6],  # x1 <-> x3
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.6, 0.0, 1.0],
        ],
        dtype=float,
    )

    n = 100
    beta_shape = 3.0

    mggd = MGGD(np.zeros(d), omega_true, beta_shape)
    eps = mggd.rvs(size=n)

    A = np.eye(d) - B_true
    X = eps @ np.linalg.inv(A)
    X -= X.mean(axis=0, keepdims=True)

    model = ABICLiNGAM(beta=beta_shape)
    model.fit(X)

    am = model.adjacency_matrix_
    cm = model.coefficient_matrix_
    ecm = model.error_covariance_matrix_


def test_fit_invalid_data():

    try:
        model = ABICLiNGAM(beta=-1.0)
    except ValueError as e:
        pass
    else:
        raise AssertionError

    try:
        model = ABICLiNGAM(lam=-0.5)
    except ValueError as e:
        pass
    else:
        raise AssertionError

    try:
        model = ABICLiNGAM(acyc_order=0)
    except ValueError as e:
        pass
    else:
        raise AssertionError

    try:
        model = ABICLiNGAM(acyc_order=1.5)
    except TypeError as e:
        pass
    else:
        raise AssertionError

    try:
        model = ABICLiNGAM(min_causal_effect=-0.1)
    except ValueError as e:
        pass
    else:
        raise AssertionError

    try:
        model = ABICLiNGAM(min_error_covariance=-0.1)
    except ValueError as e:
        pass
    else:
        raise AssertionError

    try:
        model = ABICLiNGAM(max_outer=0)
    except ValueError as e:
        pass
    else:
        raise AssertionError

    try:
        model = ABICLiNGAM(tol_h=0.0)
    except ValueError as e:
        pass
    else:
        raise AssertionError

    try:
        model = ABICLiNGAM(rho_max=0.0)
    except ValueError as e:
        pass
    else:
        raise AssertionError

    try:
        model = ABICLiNGAM(inner_start=0)
    except ValueError as e:
        pass
    else:
        raise AssertionError

    try:
        model = ABICLiNGAM(inner_growth=-1)
    except ValueError as e:
        pass
    else:
        raise AssertionError

    try:
        model = ABICLiNGAM(inner_tol=0.0)
    except ValueError as e:
        pass
    else:
        raise AssertionError

    model = ABICLiNGAM()

    try:
        model.fit(np.array([1, 2, 3]))
    except ValueError as e:
        pass
    else:
        raise AssertionError

    try:
        model.fit(np.array([[1], [2], [3]]))
    except ValueError as e:
        pass
    else:
        raise AssertionError

    try:
        model.fit(pd.DataFrame())
    except ValueError as e:
        pass
    else:
        raise AssertionError


def test_bootstrap_success():
    d = 4
    B_true = np.zeros((d, d), dtype=float)
    B_true[0, 1] = 1.0  # x0 -> x1
    B_true[1, 2] = -1.5  # x1 -> x2
    B_true[2, 3] = 1.0  # x2 -> x3

    omega_true = np.array(
        [
            [1.2, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.6],  # x1 <-> x3
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.6, 0.0, 1.0],
        ],
        dtype=float,
    )

    n = 100
    beta_shape = 3.0
    mggd = MGGD(np.zeros(d), omega_true, beta_shape)
    eps = mggd.rvs(size=n)

    A = np.eye(d) - B_true
    X = eps @ np.linalg.inv(A)
    X -= X.mean(axis=0, keepdims=True)

    model = ABICLiNGAM(beta=beta_shape)
    result = model.bootstrap(X, n_sampling=2)

    am_samples = result.adjacency_matrices_
    cm_samples = result.coefficient_matrices_
    ecm_samples = result.error_covariance_matrices_
