import os

import numpy as np
import pandas as pd

from lingam.longitudinal_lingam import LongitudinalLiNGAM


def test_fit_success():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 0.7*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.3*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.3*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.7*x1 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.5*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.5*x1 + np.random.uniform(size=1000)
    X3 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = np.empty((3, 1000, 4))
    X_list[0] = X1
    X_list[1] = X2
    X_list[2] = X3

    model = LongitudinalLiNGAM()
    model.fit(X_list)

    # check causal ordering
    cos = model.causal_orders_
    for co in cos[1:]:
        assert co.index(0) < co.index(1) < co.index(3)

    # check B(t,t)
    B_t = model.adjacency_matrices_[1, 0] # B(1,1)
    assert B_t[1, 0] > 0.1 and B_t[3, 1] > 0.5
    B_t[1, 0] = 0.0
    B_t[3, 1] = 0.0
    assert np.sum(B_t) < 0.1

    B_t = model.adjacency_matrices_[2, 0] # B(2,2)
    assert B_t[1, 0] > 0.3 and B_t[3, 1] > 0.3
    B_t[1, 0] = 0.0
    B_t[3, 1] = 0.0
    assert np.sum(B_t) < 0.1

    # check B(t,t-τ)
    B_tau = model.adjacency_matrices_[1, 1] # B(1,0)
    assert B_tau[0, 2] > 0.3 and B_tau[2, 3] > 0.3

    B_tau = model.adjacency_matrices_[1, 1] # B(2,1)
    assert B_tau[0, 2] > 0.3 and B_tau[2, 3] > 0.3

    # fit by list
    X_list = [X1, X2, X3]
    model = LongitudinalLiNGAM()
    model.fit(X_list)

    p_values = model.get_error_independence_p_values()
    resid = model.residuals_

    # prior knowledge
    pk = np.ones((3, 2, 4, 4)) * -1
    model = LongitudinalLiNGAM(prior_knowledge=pk)
    model.fit(X_list)

def test_fit_invalid_data():
    # Different features
    x0 = np.random.uniform(size=1000)
    x1 = 0.7*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.3*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.3*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    X2 = pd.DataFrame(np.array([x0, x1, x2]).T,
                      columns=['x0', 'x1', 'x2'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.5*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.5*x1 + np.random.uniform(size=1000)
    X3 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2, X3]

    try:
        model = LongitudinalLiNGAM()
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Not list data
    X = 1
    try:
        model = LongitudinalLiNGAM()
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
        model = LongitudinalLiNGAM()
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # List items
    X_list = [X1]

    try:
        model = LongitudinalLiNGAM()
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include non-numeric data
    x0 = np.random.uniform(size=1000)
    x1 = 0.7*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.3*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = np.array(['X']*1000) # <== non-numeric
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = np.array(['X']*1000) # <== non-numeric
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.5*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.5*x1 + np.random.uniform(size=1000)
    X3 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2, X3]

    try:
        model = LongitudinalLiNGAM()
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include NaN values
    x0 = np.random.uniform(size=1000)
    x1 = 0.7*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.3*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.3*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.7*x1 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])
    X2.iloc[100, 0] = np.nan # set nan

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.5*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.5*x1 + np.random.uniform(size=1000)
    X3 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2, X3]

    try:
        model = LongitudinalLiNGAM()
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include infinite values
    x0 = np.random.uniform(size=1000)
    x1 = 0.7*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.3*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.3*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.7*x1 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])
    X2.iloc[100, 0] = np.inf # set inf

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.5*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.5*x1 + np.random.uniform(size=1000)
    X3 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2, X3]

    try:
        model = LongitudinalLiNGAM()
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # prior knowledge
    pk = np.ones((3, 2, 4, 4)) * -1
    try:
        # pk.shape[1] != n_lags + 1
        model = LongitudinalLiNGAM(n_lags=10, prior_knowledge=pk)
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

def test_bootstrap_success():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 0.7*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.3*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.3*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.7*x1 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.5*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.5*x1 + np.random.uniform(size=1000)
    X3 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = np.empty((3, 1000, 4))
    X_list[0] = X1
    X_list[1] = X2
    X_list[2] = X3

    model = LongitudinalLiNGAM()
    model.bootstrap(X_list, n_sampling=3)

    # fit by list
    X_list = [X1, X2, X3]
    model = LongitudinalLiNGAM()
    result = model.bootstrap(X_list, n_sampling=3)

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

    # get_paths
    result.get_paths(0, 1, 1, 2)


def test_bootstrap_invalid_data():
    # Different features
    x0 = np.random.uniform(size=1000)
    x1 = 0.7*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.3*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.3*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    X2 = pd.DataFrame(np.array([x0, x1, x2]).T,
                      columns=['x0', 'x1', 'x2'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.5*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.5*x1 + np.random.uniform(size=1000)
    X3 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2, X3]

    try:
        model = LongitudinalLiNGAM()
        model.bootstrap(X_list, n_sampling=3)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Not list data
    X = 1
    try:
        model = LongitudinalLiNGAM()
        model.bootstrap(X, n_sampling=3)
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
        model = LongitudinalLiNGAM()
        model.bootstrap(X_list, n_sampling=3)
    except ValueError:
        pass
    else:
        raise AssertionError

    # List items
    X_list = [X1]

    try:
        model = LongitudinalLiNGAM()
        model.bootstrap(X_list, n_sampling=3)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include non-numeric data
    x0 = np.random.uniform(size=1000)
    x1 = 0.7*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.3*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = np.array(['X']*1000) # <== non-numeric
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = np.array(['X']*1000) # <== non-numeric
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.5*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.5*x1 + np.random.uniform(size=1000)
    X3 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2, X3]

    try:
        model = LongitudinalLiNGAM()
        model.bootstrap(X_list, n_sampling=3)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include NaN values
    x0 = np.random.uniform(size=1000)
    x1 = 0.7*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.3*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.3*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.7*x1 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])
    X2.iloc[100, 0] = np.nan # set nan

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.5*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.5*x1 + np.random.uniform(size=1000)
    X3 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2, X3]

    try:
        model = LongitudinalLiNGAM()
        model.bootstrap(X_list, n_sampling=3)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include infinite values
    x0 = np.random.uniform(size=1000)
    x1 = 0.7*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.3*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.3*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.7*x1 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])
    X2.iloc[100, 0] = np.inf # set inf

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.5*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.5*x1 + np.random.uniform(size=1000)
    X3 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2, X3]

    try:
        model = LongitudinalLiNGAM()
        model.bootstrap(X_list, n_sampling=3)
    except ValueError:
        pass
    else:
        raise AssertionError

    x0 = np.random.uniform(size=1000)
    x1 = np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T, columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000)
    x1 = np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T, columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000)
    x1 = np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = np.random.uniform(size=1000)
    X3 = pd.DataFrame(np.array([x0, x1, x2, x3]).T, columns=['x0', 'x1', 'x2', 'x3'])
    
    X_list = [X1, X2, X3]

    # Invalid argument: bootstrap(n_sampling=-1)
    model = LongitudinalLiNGAM()
    try:
        result = model.bootstrap(X_list, n_sampling=-1)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: get_causal_direction_counts(n_directions=-1)
    model = LongitudinalLiNGAM()
    result = model.bootstrap(X_list, n_sampling=5)
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

    # Invalid argument: get_paths(min_causal_effect<0)
    try:
        result.get_paths(0, 1, 1, 2, -1)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: get_paths(to_t < from_t)
    try:
        result.get_paths(0, 1, 1, 0)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: get_paths(to_t < from_t)
    try:
        result.get_paths(0, 1, 1, 0)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: get_paths(same variable)
    try:
        result.get_paths(0, 0, 1, 1)
    except ValueError:
        pass
    else:
        raise AssertionError

def test_estimate_total_effect_invalid():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 0.7*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.3*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.3*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.7*x1 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.5*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.5*x1 + np.random.uniform(size=1000)
    X3 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = np.empty((3, 1000, 4))
    X_list[0] = X1
    X_list[1] = X2
    X_list[2] = X3

    model = LongitudinalLiNGAM()
    model.fit(X_list)
    model._causal_orders[0] = [0, 1, 2, 3]
    model._causal_orders[1] = [0, 1, 2, 3]
    model._causal_orders[2] = [0, 1, 2, 3]

    # warning
    model.estimate_total_effect(X_list, 2, 1, 2, 0)
    model.estimate_total_effect(X_list, 2, 1, 1, 1)
