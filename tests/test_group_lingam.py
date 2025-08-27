import os

import numpy as np
import pandas as pd
from lingam.group_lingam import GroupLiNGAM

def get_external_effect(n_samples):
    zi = np.random.normal(0, 1, n_samples)
    qi = np.random.choice(np.concatenate((np.random.uniform(0.5, 0.8, n_samples//2),
                                          np.random.uniform(1.2, 2.0, n_samples//2))))
    ei = np.sign(zi) * np.abs(zi) ** qi
    ei = (ei - np.mean(ei)) / np.std(ei)
    return ei

def test_fit_success():
    # causal direction: x0, x1 --> x2, x3 --> x4
    x0 = get_external_effect(500)
    x1 = get_external_effect(500)
    x2 = -1.21 * x1 + get_external_effect(500)
    x3 = -1.98 * x0 + get_external_effect(500)
    x4 = -0.55 * x1 - 0.50 * x2 - 1.16 * x3 + get_external_effect(500)
    X = pd.DataFrame(np.array([x0, x1, x2, x3, x4]).T, columns=['x0', 'x1', 'x2', 'x3', 'x4'])

    model = GroupLiNGAM()
    model.fit(X)

    co = model.causal_order_
    am = model.adjacency_matrix_

def test_fit_invalid():
    # Not array data
    X = 1
    try:
        model = GroupLiNGAM()
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
        model = GroupLiNGAM()
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
        model = GroupLiNGAM()
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
        model = GroupLiNGAM()
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

def test_bootstrap_success():
    # causal direction: x0, x1 --> x2, x3 --> x4
    x0 = get_external_effect(500)
    x1 = get_external_effect(500)
    x2 = -1.21 * x1 + get_external_effect(500)
    x3 = -1.98 * x0 + get_external_effect(500)
    x4 = -0.55 * x1 - 0.50 * x2 - 1.16 * x3 + get_external_effect(500)
    X = pd.DataFrame(np.array([x0, x1, x2, x3, x4]).T, columns=['x0', 'x1', 'x2', 'x3', 'x4'])

    model = GroupLiNGAM()
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
    # causal direction: x0, x1 --> x2, x3 --> x4
    x0 = get_external_effect(500)
    x1 = get_external_effect(500)
    x2 = -1.21 * x1 + get_external_effect(500)
    x3 = -1.98 * x0 + get_external_effect(500)
    x4 = -0.55 * x1 - 0.50 * x2 - 1.16 * x3 + get_external_effect(500)
    X = pd.DataFrame(np.array([x0, x1, x2, x3, x4]).T, columns=['x0', 'x1', 'x2', 'x3', 'x4'])

    # Invalid argument: bootstrap(n_sampling=-1)
    model = GroupLiNGAM()
    try:
        result = model.bootstrap(X, n_sampling=-1)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: bootstrap(n_sampling='3')
    model = GroupLiNGAM()
    try:
        result = model.bootstrap(X, n_sampling='3')
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: get_causal_direction_counts(n_directions=-1)
    model = GroupLiNGAM()
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
