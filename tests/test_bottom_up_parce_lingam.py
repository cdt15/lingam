import os

import numpy as np
import pandas as pd
from lingam.bottom_up_parce_lingam import BottomUpParceLiNGAM
from sklearn.linear_model import LinearRegression


def test_fit_success():
    # causal direction: x5 --> x1, x3 --> x0 --> x2 --> x4
    n_samples = 300
    x5 = np.random.uniform(size=n_samples)
    x1 = 2.0 * x5 + np.random.uniform(size=n_samples)
    x3 = 2.0 * x5 + np.random.uniform(size=n_samples)
    x0 = 0.5 * x1 - 0.5 * x3 + np.random.uniform(size=n_samples)
    x2 = 0.5 * x0 + np.random.uniform(size=n_samples)
    x4 = 1.0 * x2 + np.random.uniform(size=n_samples)
    # x5 is latent confounders
    X = pd.DataFrame(
        np.array([x0, x1, x2, x3, x4]).T, columns=['x0', 'x1', 'x2', 'x3', 'x4']
    )

    model = BottomUpParceLiNGAM()
    model.fit(X)

    co = model.causal_order_
    am = model.adjacency_matrix_

    te = model.estimate_total_effect(X, 0, 4)
    # assert te > 0.3 and te < 0.7

    te = model.estimate_total_effect(X, 4, 0)
    te = model.estimate_total_effect(X, 4, 2)

    p_values = model.get_error_independence_p_values(X)
    # assert p_values.shape[0] == 5 and p_values.shape[1] == 5

    # reject
    model = BottomUpParceLiNGAM(alpha=1.0)
    model.fit(X)
    te = model.estimate_total_effect(X, 4, 0)
    p_values = model.get_error_independence_p_values(X)

    # coverage
    model = BottomUpParceLiNGAM(alpha=0.0)
    model.fit(X)
    model._adjacency_matrix[0, 1] = np.nan
    model._adjacency_matrix[1, 0] = np.nan
    p_values = model.get_error_independence_p_values(X)

    model = BottomUpParceLiNGAM(alpha=0.0)
    model.fit(X)
    model._causal_order = [[0, 1], 2, 3, 4]
    model.estimate_total_effect(X, 0, 1)

    # f-correlation
    model = BottomUpParceLiNGAM(independence="fcorr")
    model.fit(X)


def test_bootstrap_success():
    # causal direction: x5 --> x1, x3 --> x0 --> x2 --> x4
    n_samples = 300
    x5 = np.random.uniform(size=n_samples)
    x1 = 2.0 * x5 + np.random.uniform(size=n_samples)
    x3 = 2.0 * x5 + np.random.uniform(size=n_samples)
    x0 = 0.5 * x1 - 0.5 * x3 + np.random.uniform(size=n_samples)
    x2 = 0.5 * x0 + np.random.uniform(size=n_samples)
    x4 = 1.0 * x2 + np.random.uniform(size=n_samples)
    # x5 is latent confounders
    X = pd.DataFrame(
        np.array([x0, x1, x2, x3, x4]).T, columns=['x0', 'x1', 'x2', 'x3', 'x4']
    )

    model = BottomUpParceLiNGAM()
    result = model.bootstrap(X, n_sampling=5)
    am = result.adjacency_matrices_
    assert len(am) == 5
    te = result.total_effects_
    assert len(te) == 5

    cdc = result.get_causal_direction_counts()
    dagc = result.get_directed_acyclic_graph_counts()
    probs = result.get_probabilities()
    ce = result.get_total_causal_effects()
    paths = result.get_paths(3, 4)

    # f-correlation
    model = BottomUpParceLiNGAM(independence="fcorr")
    result = model.bootstrap(X, n_sampling=5)


def test_fit_invalid_data():
    # Not array data
    X = 1
    try:
        model = BottomUpParceLiNGAM()
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include non-numeric data
    x0 = np.random.uniform(size=5)
    x1 = np.array(['X', 'Y', 'X', 'Y', 'X'])
    X = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])
    try:
        model = BottomUpParceLiNGAM()
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include NaN values
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])
    X.iloc[100, 0] = np.nan
    try:
        model = BottomUpParceLiNGAM()
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include infinite values
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])
    X.iloc[100, 0] = np.inf
    try:
        model = BottomUpParceLiNGAM()
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # invalid regressor
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])
    try:
        model = BottomUpParceLiNGAM(regressor=1)
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # invalid alpha
    try:
        model = BottomUpParceLiNGAM(alpha=-1)
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid value: independence
    try:
        model = BottomUpParceLiNGAM(independence="lingam")
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # invalid ind_corr
    try:
        model = BottomUpParceLiNGAM(ind_corr=-1.0)
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError


def test_prior_knowledge_success():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=300)
    x1 = 2.0 * x0 + np.random.uniform(size=300)
    x2 = np.random.uniform(size=300)
    x3 = 4.0 * x1 + np.random.uniform(size=300)
    X = pd.DataFrame(np.array([x0, x1, x2, x3]).T, columns=['x0', 'x1', 'x2', 'x3'])

    # prior knowledge: nothing
    pk = np.array(
        [
            [0, -1, -1, -1],
            [-1, 0, -1, -1],
            [-1, -1, 0, -1],
            [-1, -1, -1, 0],
        ]
    )

    model = BottomUpParceLiNGAM(prior_knowledge=pk)
    model.fit(X)
    co = model.causal_order_
    print(co)

    # prior knowledge: x1 is exogenous
    pk = np.array(
        [
            [0, -1, -1, -1],
            [0, 0, 0, 0],
            [-1, -1, 0, -1],
            [-1, -1, -1, 0],
        ]
    )

    model = BottomUpParceLiNGAM(prior_knowledge=pk)
    model.fit(X)
    co = model.causal_order_
    print(co)

    # prior knowledge: x0 is sink
    pk = np.array(
        [
            [0, -1, -1, -1],
            [0, 0, -1, -1],
            [0, -1, 0, -1],
            [0, -1, -1, 0],
        ]
    )

    model = BottomUpParceLiNGAM(prior_knowledge=pk)
    model.fit(X)
    co = model.causal_order_
    print(co)

    # prior knowledge: x2-->x3 has path
    pk = np.array(
        [
            [0, -1, -1, -1],
            [-1, 0, -1, -1],
            [-1, -1, 0, 0],
            [-1, -1, 1, 0],
        ]
    )

    model = BottomUpParceLiNGAM(prior_knowledge=pk)
    model.fit(X)
    co = model.causal_order_
    print(co)

    # prior knowledge: x1-->x3 does not have path
    pk = np.array(
        [
            [0, -1, -1, -1],
            [-1, 0, -1, -1],
            [-1, -1, 0, -1],
            [-1, 0, -1, 0],
        ]
    )

    model = BottomUpParceLiNGAM(prior_knowledge=pk)
    model.fit(X)


def test_prior_knowledge_invalid():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0 * x1 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3]).T, columns=['x0', 'x1', 'x2', 'x3'])

    # prior knowledge: invalid
    pk = np.array(
        [
            [0, -1, 1],
            [-1, 0, -1],
            [-1, -1, 0],
            [-1, -1, -1],
        ]
    )

    try:
        model = BottomUpParceLiNGAM(prior_knowledge=pk)
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # prior knowledge: inconsistent
    pk = np.array(
        [
            [0, 1, -1, -1],
            [1, 0, -1, -1],
            [-1, -1, 0, -1],
            [-1, -1, -1, 0],
        ]
    )

    try:
        model = BottomUpParceLiNGAM(prior_knowledge=pk)
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError


def test_fit_regressor():
    # causal direction: x5 --> x1, x3 --> x0 --> x2 --> x4
    n_samples = 300
    x5 = np.random.uniform(size=n_samples)
    x1 = 2.0 * x5 + np.random.uniform(size=n_samples)
    x3 = 2.0 * x5 + np.random.uniform(size=n_samples)
    x0 = 0.5 * x1 - 0.5 * x3 + np.random.uniform(size=n_samples)
    x2 = 0.5 * x0 + np.random.uniform(size=n_samples)
    x4 = 1.0 * x2 + np.random.uniform(size=n_samples)
    # x5 is latent confounders
    X = pd.DataFrame(
        np.array([x0, x1, x2, x3, x4]).T, columns=['x0', 'x1', 'x2', 'x3', 'x4']
    )

    reg = LinearRegression()

    model = BottomUpParceLiNGAM(regressor=reg)
    model.fit(X)


def test_bootstrap_invalid_data():
    x0 = np.random.uniform(size=300)
    x1 = np.random.uniform(size=300)
    x2 = np.random.uniform(size=300)
    x3 = np.random.uniform(size=300)
    X_success = pd.DataFrame(
        np.array([x0, x1, x2, x3]).T, columns=['x0', 'x1', 'x2', 'x3']
    )

    # Invalid argument: bootstrap(n_sampling=-1)
    model = BottomUpParceLiNGAM()
    try:
        result = model.bootstrap(X_success, n_sampling=-1)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: bootstrap(n_sampling='3')
    model = BottomUpParceLiNGAM()
    try:
        result = model.bootstrap(X_success, n_sampling='3')
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: get_causal_direction_counts(n_directions=-1)
    model = BottomUpParceLiNGAM()
    result = model.bootstrap(X_success, n_sampling=5)
    try:
        result.get_causal_direction_counts(n_directions=-1)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: get_causal_direction_counts(n_directions='3')
    model = BottomUpParceLiNGAM()
    result = model.bootstrap(X_success, n_sampling=5)
    try:
        result.get_causal_direction_counts(n_directions='3')
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: get_causal_direction_counts(min_causal_effect=-1.0)
    try:
        result.get_causal_direction_counts(min_causal_effect=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: get_directed_acyclic_graph_counts(n_dags=-1)
    try:
        result.get_directed_acyclic_graph_counts(n_dags=-1)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: get_directed_acyclic_graph_counts(n_dags='3')
    try:
        result.get_directed_acyclic_graph_counts(n_dags='3')
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: get_directed_acyclic_graph_counts(min_causal_effect=-1.0)
    try:
        result.get_directed_acyclic_graph_counts(min_causal_effect=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: get_probabilities(min_causal_effect=-1.0)
    try:
        result.get_probabilities(min_causal_effect=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: get_total_causal_effects(min_causal_effect=-1.0)
    try:
        result.get_total_causal_effects(min_causal_effect=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: get_paths(min_causal_effect=-1.0)
    try:
        result.get_paths(0, 1, min_causal_effect=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError
