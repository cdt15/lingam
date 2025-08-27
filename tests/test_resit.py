import os

import numpy as np
import pandas as pd
from lingam.resit import RESIT
from sklearn.ensemble import RandomForestRegressor


def test_fit_success():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0 * x1 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3]).T, columns=["x0", "x1", "x2", "x3"])

    reg = RandomForestRegressor(max_depth=4, random_state=0)
    model = RESIT(regressor=reg)
    model.fit(X)

    # check the causal ordering
    co = model.causal_order_

    # check the adjacency matrix
    am = model.adjacency_matrix_

    # check estimate_total_effect (Not implement)
    te = model.estimate_total_effect(X, 0, 3)

    # check get_error_independence_p_values (Not implement)
    p_values = model.get_error_independence_p_values(X)


def test_fit_invalid():
    reg = RandomForestRegressor(max_depth=4, random_state=0)

    # Not array data
    X = 1
    try:
        model = RESIT(regressor=reg)
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0 * x1 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3]).T, columns=["x0", "x1", "x2", "x3"])

    # Invalid regressor
    try:
        model = RESIT(regressor=None)
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid regressor
    dummy_reg = lambda x: x
    try:
        model = RESIT(regressor=dummy_reg)
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid alpha
    try:
        model = RESIT(regressor=reg, alpha=-1)
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError


def test_bootstrap_success():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0 * x1 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3]).T, columns=["x0", "x1", "x2", "x3"])

    reg = RandomForestRegressor(max_depth=4, random_state=0)
    model = RESIT(regressor=reg)
    result = model.bootstrap(X, n_sampling=5)
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
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0 * x1 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3]).T, columns=["x0", "x1", "x2", "x3"])

    reg = RandomForestRegressor(max_depth=4, random_state=0)

    # Invalid argument: bootstrap(n_sampling=-1)
    model = RESIT(regressor=reg)
    try:
        result = model.bootstrap(X, n_sampling=-1)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: bootstrap(n_sampling='3')
    model = RESIT(regressor=reg)
    try:
        result = model.bootstrap(X, n_sampling='3')
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: get_causal_direction_counts(n_directions=-1)
    model = RESIT(regressor=reg)
    result = model.bootstrap(X, n_sampling=5)
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
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0 * x1 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3]).T, columns=["x0", "x1", "x2", "x3"])

    reg = RandomForestRegressor(max_depth=4, random_state=0)

    # prior knowledge: nothing
    pk = np.array(
        [
            [0, -1, -1, -1],
            [-1, 0, -1, -1],
            [-1, -1, 0, -1],
            [-1, -1, -1, 0],
        ]
    )
    model = RESIT(regressor=reg, prior_knowledge=pk)
    model.fit(X)
    co = model.causal_order_

    # prior knowledge: x1 is exogenous
    pk = np.array(
        [
            [0, -1, -1, -1],
            [0, 0, 0, 0],
            [-1, -1, 0, -1],
            [-1, -1, -1, 0],
        ]
    )
    model = RESIT(regressor=reg, prior_knowledge=pk)
    model.fit(X)
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
    model = RESIT(regressor=reg, prior_knowledge=pk)
    model.fit(X)
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
    model = RESIT(regressor=reg, prior_knowledge=pk)
    model.fit(X)
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
    model = RESIT(regressor=reg, prior_knowledge=pk)
    model.fit(X)
    co = model.causal_order_


def test_prior_knowledge_invalid():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0 * x1 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3]).T, columns=["x0", "x1", "x2", "x3"])

    reg = RandomForestRegressor(max_depth=4, random_state=0)

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
        model = RESIT(regressor=reg, prior_knowledge=pk)
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
        model = RESIT(regressor=reg, prior_knowledge=pk)
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError
