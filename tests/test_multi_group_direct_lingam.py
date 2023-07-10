import os

import numpy as np
import pandas as pd

from lingam.multi_group_direct_lingam import MultiGroupDirectLiNGAM


def test_fit_success():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=500)
    x1 = 2.1*x0 + np.random.uniform(size=500)
    x2 = np.random.uniform(size=500)
    x3 = 4.2*x1 + np.random.uniform(size=500)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2]

    model = MultiGroupDirectLiNGAM()
    model.fit(X_list)

    # check the causal ordering
    co = model.causal_order_
    assert co.index(0) < co.index(1) < co.index(3)

    # check the adjacency matrix
    am = model.adjacency_matrices_[0]
    # assert am[1, 0] > 1.5 and am[3, 1] > 3.5

    # check the adjacency matrix
    am = model.adjacency_matrices_[1]
    # assert am[1, 0] > 1.5 and am[3, 1] > 3.5

    am[1, 0] = 0.0
    am[3, 1] = 0.0
    # assert np.sum(am) < 0.1

    co = model.causal_order_
    am = model.adjacency_matrices_
    p_values = model.get_error_independence_p_values(X_list)


def test_fit_invalid_data():
    # Different features
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000)
    x1 = 2.1*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2]).T,
                      columns=['x0', 'x1', 'x2'])

    X_list = [X1, X2]

    try:
        model = MultiGroupDirectLiNGAM()
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Not list data
    X = 1
    try:
        model = MultiGroupDirectLiNGAM()
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include not-array data
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, 1]

    try:
        model = MultiGroupDirectLiNGAM()
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # 
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1]

    try:
        model = MultiGroupDirectLiNGAM()
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include non-numeric data
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=5)
    x1 = np.array(['X', 'Y', 'X', 'Y', 'X'])
    X2 = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])

    X_list = [X1, X2]

    try:
        model = MultiGroupDirectLiNGAM()
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include NaN values
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])
    X2.iloc[100, 0] = np.nan

    X_list = [X1, X2]

    try:
        model = MultiGroupDirectLiNGAM()
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include infinite values
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])
    X.iloc[100, 0] = np.inf

    X_list = [X1, X2]

    try:
        model = MultiGroupDirectLiNGAM()
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError


def test_prior_knowledge_success():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000)
    x1 = 2.1*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.2*x1 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2]

    # prior knowledge: nothing
    pk = np.array([
        [0, -1, -1, -1],
        [-1,  0, -1, -1],
        [-1, -1,  0, -1],
        [-1, -1, -1,  0],
    ])

    model = MultiGroupDirectLiNGAM(prior_knowledge=pk)
    model.fit(X_list)
    co = model.causal_order_
    assert co.index(0) < co.index(1) < co.index(3)

    # prior knowledge: x1 is exogenous
    pk = np.array([
        [0, -1, -1, -1],
        [0,  0,  0,  0],
        [-1, -1,  0, -1],
        [-1, -1, -1,  0],
    ])

    model = MultiGroupDirectLiNGAM(prior_knowledge=pk)
    model.fit(X_list)
    co = model.causal_order_
    print(co)
    assert co.index(1) < co.index(0)
    assert co.index(1) < co.index(2)
    assert co.index(1) < co.index(3)

    # prior knowledge: x0 is sink
    pk = np.array([
        [0, -1, -1, -1],
        [0,  0, -1, -1],
        [0, -1,  0, -1],
        [0, -1, -1,  0],
    ])

    model = MultiGroupDirectLiNGAM(prior_knowledge=pk)
    model.fit(X_list)
    co = model.causal_order_
    print(co)
    assert co.index(0) > co.index(1)
    assert co.index(0) > co.index(2)
    assert co.index(0) > co.index(3)

    # prior knowledge: x2-->x3 has path
    pk = np.array([
        [0, -1, -1, -1],
        [-1,  0, -1, -1],
        [-1, -1,  0,  0],
        [-1, -1,  1,  0],
    ])

    model = MultiGroupDirectLiNGAM(prior_knowledge=pk)
    model.fit(X_list)
    co = model.causal_order_
    print(co)
    assert co.index(2) < co.index(3)

    # prior knowledge: x1-->x3 does not have path
    pk = np.array([
        [0, -1, -1, -1],
        [-1,  0, -1, -1],
        [-1, -1,  0, -1],
        [-1,  0, -1,  0],
    ])

    model = MultiGroupDirectLiNGAM(prior_knowledge=pk)
    model.fit(X_list)


def test_prior_knowledge_invalid():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000)
    x1 = 2.1*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.2*x1 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2]

    # prior knowledge: invalid
    pk = np.array([
        [0, -1, 1],
        [-1,  0, -1],
        [-1, -1,  0],
        [-1, -1, -1],
    ])

    try:
        model = MultiGroupDirectLiNGAM(prior_knowledge=pk)
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

def test_bootstrap_success():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000)
    x1 = 2.1*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.2*x1 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2]

    model = MultiGroupDirectLiNGAM()
    result_list = model.bootstrap(X_list, n_sampling=5)

    result = result_list[0]
    result.adjacency_matrices_
    result.total_effects_

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

def test_bootstrap_invalid_data():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000)
    x1 = 2.1*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.2*x1 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2]

    # Invalid argument: bootstrap(n_sampling=-1)
    model = MultiGroupDirectLiNGAM()
    try:
        result = model.bootstrap(X_list, n_sampling=-1)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: bootstrap(n_sampling='3')
    model = MultiGroupDirectLiNGAM()
    try:
        result = model.bootstrap(X_list, n_sampling='3')
    except ValueError:
        pass
    else:
        raise AssertionError

def test_estimate_total_effect_invalid():
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000)
    x1 = 2.1*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.2*x1 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2]

    model = MultiGroupDirectLiNGAM()
    model.fit(X_list)
    model._causal_order = [0, 1, 2, 3]

    # warning
    model.estimate_total_effect(X_list, 3, 1)
