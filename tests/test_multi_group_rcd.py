import os
import random

import numpy as np
import pandas as pd

from lingam import MultiGroupRCD


def test_fit_success():
    def get_coef():
        coef = random.random()
        return coef if coef >= 0.5 else coef - 1.0

    get_external_effect = lambda n: np.random.normal(0.0, 0.5, n) ** 3

    B1 = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, get_coef(), 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, get_coef(), 0.0],
            [get_coef(), get_coef(), 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, get_coef(), 0.0, 0.0, 0.0, get_coef()],
            [0.0, 0.0, get_coef(), 0.0, 0.0, 0.0, get_coef()],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    samples = 500
    x5 = get_external_effect(samples)
    x6 = get_external_effect(samples)
    x0 = x5 * B1[0, 5] + get_external_effect(samples)
    x1 = x5 * B1[1, 5] + get_external_effect(samples)
    x2 = x0 * B1[2, 0] + x1 * B1[2, 1] + get_external_effect(samples)
    x3 = x2 * B1[3, 2] + x6 * B1[3, 6] + get_external_effect(samples)
    x4 = x2 * B1[4, 2] + x6 * B1[4, 6] + get_external_effect(samples)

    # x5, x6 is a latent variable.
    X1 = pd.DataFrame(
        np.array([x0, x1, x2, x3, x4, x5]).T,
        columns=["x0", "x1", "x2", "x3", "x4", "x5"],
    )

    B2 = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, get_coef(), 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, get_coef(), 0.0],
            [get_coef(), get_coef(), 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, get_coef(), 0.0, 0.0, 0.0, get_coef()],
            [0.0, 0.0, get_coef(), 0.0, 0.0, 0.0, get_coef()],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    samples = 1000
    x5 = get_external_effect(samples)
    x6 = get_external_effect(samples)
    x0 = x5 * B2[0, 5] + get_external_effect(samples)
    x1 = x5 * B2[1, 5] + get_external_effect(samples)
    x2 = x0 * B2[2, 0] + x1 * B2[2, 1] + get_external_effect(samples)
    x3 = x2 * B2[3, 2] + x6 * B2[3, 6] + get_external_effect(samples)
    x4 = x2 * B2[4, 2] + x6 * B2[4, 6] + get_external_effect(samples)

    # x5, x6 is a latent variable.
    X2 = pd.DataFrame(
        np.array([x0, x1, x2, x3, x4, x5]).T,
        columns=["x0", "x1", "x2", "x3", "x4", "x5"],
    )

    X_list = [X1, X2]

    model = MultiGroupRCD(
        max_explanatory_num=2,
        cor_alpha=0.01,
        ind_alpha=0.01,
        shapiro_alpha=0.01,
        MLHSICR=True,
        bw_method="mdbs",
    )

    model.fit(X_list)

    effects = model.estimate_total_effect(X_list, from_index=2, to_index=4)
    p_values = model.get_error_independence_p_values(X_list)
    model.adjacency_matrices_
    model.ancestors_list_

    # f-correlation
    model = MultiGroupRCD(
        independence="fcorr",
    )
    model.fit(X_list)


def test_fit_invalid_data():
    def get_coef():
        coef = random.random()
        return coef if coef >= 0.5 else coef - 1.0

    get_external_effect = lambda n: np.random.normal(0.0, 0.5, n) ** 3

    B1 = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, get_coef(), 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, get_coef(), 0.0],
            [get_coef(), get_coef(), 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, get_coef(), 0.0, 0.0, 0.0, get_coef()],
            [0.0, 0.0, get_coef(), 0.0, 0.0, 0.0, get_coef()],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    samples = 500
    x5 = get_external_effect(samples)
    x6 = get_external_effect(samples)
    x0 = x5 * B1[0, 5] + get_external_effect(samples)
    x1 = x5 * B1[1, 5] + get_external_effect(samples)
    x2 = x0 * B1[2, 0] + x1 * B1[2, 1] + get_external_effect(samples)
    x3 = x2 * B1[3, 2] + x6 * B1[3, 6] + get_external_effect(samples)
    x4 = x2 * B1[4, 2] + x6 * B1[4, 6] + get_external_effect(samples)

    # x5, x6 is a latent variable.
    X1 = pd.DataFrame(
        np.array([x0, x1, x2, x3, x4, x5]).T,
        columns=["x0", "x1", "x2", "x3", "x4", "x5"],
    )

    B2 = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, get_coef(), 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, get_coef(), 0.0],
            [get_coef(), get_coef(), 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, get_coef(), 0.0, 0.0, 0.0, get_coef()],
            [0.0, 0.0, get_coef(), 0.0, 0.0, 0.0, get_coef()],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    samples = 500
    x5 = get_external_effect(samples)
    x6 = get_external_effect(samples)
    x0 = x5 * B2[0, 5] + get_external_effect(samples)
    x1 = x5 * B2[1, 5] + get_external_effect(samples)
    x2 = x0 * B2[2, 0] + x1 * B2[2, 1] + get_external_effect(samples)
    x3 = x2 * B2[3, 2] + x6 * B2[3, 6] + get_external_effect(samples)
    x4 = x2 * B2[4, 2] + x6 * B2[4, 6] + get_external_effect(samples)

    # x5, x6 is a latent variable.
    X2 = pd.DataFrame(
        np.array([x0, x1, x2, x3, x4, x5]).T,
        columns=["x0", "x1", "x2", "x3", "x4", "x5"],
    )

    X_list = [X1, X2]

    # Not array data
    X = 1
    try:
        model = MultiGroupRCD()
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid value: max_explanatory_num
    try:
        model = MultiGroupRCD(max_explanatory_num=-1)
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid value: cor_alpha
    try:
        model = MultiGroupRCD(cor_alpha=-1)
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid value: ind_alpha
    try:
        model = MultiGroupRCD(ind_alpha=-1)
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid value: shapiro_alpha
    try:
        model = MultiGroupRCD(shapiro_alpha=-1)
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid value: bw_method
    try:
        model = MultiGroupRCD(bw_method="X")
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid value: independence
    try:
        model = MultiGroupRCD(independence="lingam")
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid value: ind_corr
    try:
        model = MultiGroupRCD(ind_corr=-0.5)
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError


def test_bootstrap_success():
    def get_coef():
        coef = random.random()
        return coef if coef >= 0.5 else coef - 1.0

    get_external_effect = lambda n: np.random.normal(0.0, 0.5, n) ** 3

    B1 = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, get_coef(), 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, get_coef(), 0.0],
            [get_coef(), get_coef(), 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, get_coef(), 0.0, 0.0, 0.0, get_coef()],
            [0.0, 0.0, get_coef(), 0.0, 0.0, 0.0, get_coef()],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    samples = 500
    x5 = get_external_effect(samples)
    x6 = get_external_effect(samples)
    x0 = x5 * B1[0, 5] + get_external_effect(samples)
    x1 = x5 * B1[1, 5] + get_external_effect(samples)
    x2 = x0 * B1[2, 0] + x1 * B1[2, 1] + get_external_effect(samples)
    x3 = x2 * B1[3, 2] + x6 * B1[3, 6] + get_external_effect(samples)
    x4 = x2 * B1[4, 2] + x6 * B1[4, 6] + get_external_effect(samples)

    # x5, x6 is a latent variable.
    X1 = pd.DataFrame(
        np.array([x0, x1, x2, x3, x4, x5]).T,
        columns=["x0", "x1", "x2", "x3", "x4", "x5"],
    )

    B2 = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, get_coef(), 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, get_coef(), 0.0],
            [get_coef(), get_coef(), 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, get_coef(), 0.0, 0.0, 0.0, get_coef()],
            [0.0, 0.0, get_coef(), 0.0, 0.0, 0.0, get_coef()],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    samples = 500
    x5 = get_external_effect(samples)
    x6 = get_external_effect(samples)
    x0 = x5 * B2[0, 5] + get_external_effect(samples)
    x1 = x5 * B2[1, 5] + get_external_effect(samples)
    x2 = x0 * B2[2, 0] + x1 * B2[2, 1] + get_external_effect(samples)
    x3 = x2 * B2[3, 2] + x6 * B2[3, 6] + get_external_effect(samples)
    x4 = x2 * B2[4, 2] + x6 * B2[4, 6] + get_external_effect(samples)

    # x5, x6 is a latent variable.
    X2 = pd.DataFrame(
        np.array([x0, x1, x2, x3, x4, x5]).T,
        columns=["x0", "x1", "x2", "x3", "x4", "x5"],
    )

    X_list = [X1, X2]

    model = MultiGroupRCD(
        max_explanatory_num=2,
        cor_alpha=0.01,
        ind_alpha=0.01,
        shapiro_alpha=0.01,
        MLHSICR=True,
        bw_method="mdbs",
    )

    results = model.bootstrap(X_list, n_sampling=2)
    cdc = results[0].get_causal_direction_counts(n_directions=8, min_causal_effect=0.01)
    dagc = results[0].get_directed_acyclic_graph_counts(
        n_dags=3, min_causal_effect=0.01
    )
    prob = results[0].get_probabilities(min_causal_effect=0.01)
    effects = results[0].get_total_causal_effects(min_causal_effect=0.01)
    paths = results[0].get_paths(from_index=3, to_index=1)
