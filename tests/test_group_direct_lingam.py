import os

import numpy as np
import pandas as pd
from lingam.group_direct_lingam import GroupDirectLiNGAM


def test_fit_success():
    # causal direction: x0, x1 --> x2 --> x3, x4
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    x2 = 0.5 * x1 + np.random.uniform(-1, 1, size=1000)
    x3 = 0.7 * x2 + np.random.uniform(-2, 2, size=1000)
    x4 = 1.5 * x0 + 0.8 * x3 + np.random.uniform(-2, 2, size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3, x4]).T, columns=['x0', 'x1', 'x2', 'x3', 'x4'])
    groups = [[0, 1], [2], [3, 4]]

    model = GroupDirectLiNGAM()
    model.fit(X, groups)

    co = model.causal_order_
    am = model.adjacency_matrix_

def test_fit_invalid():
    # Not array data
    X = 1
    groups = [[0, 1], [2], [3, 4]]
    try:
        model = GroupDirectLiNGAM()
        model.fit(X, groups)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include non-numeric data
    x0 = np.random.uniform(size=5)
    x1 = np.array(['X', 'Y', 'X', 'Y', 'X'])
    X = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])
    groups = [[0], [1]]
    try:
        model = GroupDirectLiNGAM()
        model.fit(X, groups)
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
        model = GroupDirectLiNGAM()
        model.fit(X, groups)
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
        model = GroupDirectLiNGAM()
        model.fit(X, groups)
    except ValueError:
        pass
    else:
        raise AssertionError

def test_bootstrap_success():
    # causal direction: x0, x1 --> x2 --> x3, x4
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    x2 = 0.5 * x1 + np.random.uniform(-1, 1, size=1000)
    x3 = 0.7 * x2 + np.random.uniform(-2, 2, size=1000)
    x4 = 1.5 * x0 + 0.8 * x3 + np.random.uniform(-2, 2, size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3, x4]).T, columns=['x0', 'x1', 'x2', 'x3', 'x4'])
    groups = [[0, 1], [2], [3, 4]]

    model = GroupDirectLiNGAM()
    result = model.bootstrap(X, groups, n_sampling=5)
    am = result.adjacency_matrices_
    assert len(am) == 5
    te = result.total_effects_
    assert len(te) == 5

    # No argument
    cdc = result.get_causal_direction_counts()

    # n_directions=2
    cdc = result.get_causal_direction_counts(n_directions=2)

    # min_causal_effect=0.2
    cdc = result.get_causal_direction_counts(min_causal_effect=0.2)

    # split_by_causal_effect_sign=True
    cdc = result.get_causal_direction_counts(split_by_causal_effect_sign=True)

    # No argument
    dagc = result.get_directed_acyclic_graph_counts()

    # n_dags=2
    dagc = result.get_directed_acyclic_graph_counts(n_dags=2)

    # min_causal_effect=0.6
    dagc = result.get_directed_acyclic_graph_counts(min_causal_effect=0.6)

    # split_by_causal_effect_sign=True
    dagc = result.get_directed_acyclic_graph_counts(split_by_causal_effect_sign=True)

    # get_probabilities
    probs = result.get_probabilities()

    # get_probabilities
    probs = result.get_probabilities(min_causal_effect=0.6)

    # get_total_causal_effects
    ce = result.get_total_causal_effects()

    # get_total_causal_effects
    ce = result.get_total_causal_effects(min_causal_effect=0.6)

    # get_paths
    paths = result.get_paths(1, 2)

def test_bootstrap_invalid_data():
    # causal direction: x0, x1 --> x2 --> x3, x4
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    x2 = 0.5 * x1 + np.random.uniform(-1, 1, size=1000)
    x3 = 0.7 * x2 + np.random.uniform(-2, 2, size=1000)
    x4 = 1.5 * x0 + 0.8 * x3 + np.random.uniform(-2, 2, size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3, x4]).T, columns=['x0', 'x1', 'x2', 'x3', 'x4'])
    groups = [[0, 1], [2], [3, 4]]

    # Invalid argument: bootstrap(n_sampling=-1)
    model = GroupDirectLiNGAM()
    try:
        result = model.bootstrap(X, groups, n_sampling=-1)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: bootstrap(n_sampling='3')
    model = GroupDirectLiNGAM()
    try:
        result = model.bootstrap(X, groups, n_sampling='3')
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: get_causal_direction_counts(n_directions=-1)
    model = GroupDirectLiNGAM()
    result = model.bootstrap(X, groups, n_sampling=5)
    try:
        result.get_causal_direction_counts(n_directions=-1)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: get_causal_direction_counts(n_directions='3')
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

def test_prior_knowledge_success():
    # causal direction: x0, x1 --> x2 --> x3, x4
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    x2 = 0.5 * x1 + np.random.uniform(-1, 1, size=1000)
    x3 = 0.7 * x2 + np.random.uniform(-2, 2, size=1000)
    x4 = 1.5 * x0 + 0.8 * x3 + np.random.uniform(-2, 2, size=1000)
    x5 = -0.6 * x3 - 0.5 * x4 + np.random.uniform(-3, 3, size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T, columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])
    groups = [[0, 1], [2], [3, 4], [5]]

    # prior knowledge: nothing
    pk = np.array(
        [
            [0, -1, -1, -1],
            [-1, 0, -1, -1],
            [-1, -1, 0, -1],
            [-1, -1, -1, 0],
        ]
    )

    model = GroupDirectLiNGAM(prior_knowledge=pk)
    model.fit(X, groups)

    co = model.causal_order_
    assert co.index(0) < co.index(1) < co.index(3)

    # prior knowledge: x1 is exogenous
    pk = np.array(
        [
            [0, -1, -1, -1],
            [0, 0, 0, 0],
            [-1, -1, 0, -1],
            [-1, -1, -1, 0],
        ]
    )

    model = GroupDirectLiNGAM(prior_knowledge=pk)
    model.fit(X, groups)
    co = model.causal_order_
    assert co.index(1) < co.index(0)
    assert co.index(1) < co.index(2)
    assert co.index(1) < co.index(3)

    # prior knowledge: x0 is sink
    pk = np.array(
        [
            [0, -1, -1, -1],
            [0, 0, -1, -1],
            [0, -1, 0, -1],
            [0, -1, -1, 0],
        ]
    )

    model = GroupDirectLiNGAM(prior_knowledge=pk)
    model.fit(X, groups)
    co = model.causal_order_
    assert co.index(0) > co.index(1)
    assert co.index(0) > co.index(2)
    assert co.index(0) > co.index(3)

    # prior knowledge: x2-->x3 has path
    pk = np.array(
        [
            [0, -1, -1, -1],
            [-1, 0, -1, -1],
            [-1, -1, 0, 0],
            [-1, -1, 1, 0],
        ]
    )

    model = GroupDirectLiNGAM(prior_knowledge=pk)
    model.fit(X, groups)
    co = model.causal_order_
    assert co.index(2) < co.index(3)

    # prior knowledge: x1-->x3 does not have path
    pk = np.array(
        [
            [0, -1, -1, -1],
            [-1, 0, -1, -1],
            [-1, -1, 0, -1],
            [-1, 0, -1, 0],
        ]
    )

    model = GroupDirectLiNGAM(prior_knowledge=pk)
    model.fit(X, groups)

def test_prior_knowledge_invalid():
    # causal direction: x0, x1 --> x2 --> x3, x4
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    x2 = 0.5 * x1 + np.random.uniform(-1, 1, size=1000)
    x3 = 0.7 * x2 + np.random.uniform(-2, 2, size=1000)
    x4 = 1.5 * x0 + 0.8 * x3 + np.random.uniform(-2, 2, size=1000)
    x5 = -0.6 * x3 - 0.5 * x4 + np.random.uniform(-3, 3, size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T, columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])
    groups = [[0, 1], [2], [3, 4], [5]]

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
        model = GroupDirectLiNGAM(prior_knowledge=pk)
        model.fit(X, groups)
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
        model = GroupDirectLiNGAM(prior_knowledge=pk)
        model.fit(X, groups)
    except ValueError:
        pass
    else:
        raise AssertionError
