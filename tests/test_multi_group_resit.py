import numpy as np
import pandas as pd
import pytest
from lingam import MultiGroupRESIT
from sklearn.ensemble import RandomForestRegressor


def create_data(n=100, seed=0):
    np.random.seed(seed)

    # causal direction: x0 --> x1 --> x3
    X_list = []
    for _ in range(3):
        x0 = np.random.uniform(size=n)
        x1 = 2.0 * x0 + np.random.uniform(size=n)
        x2 = np.random.uniform(size=n)
        x3 = 4.0 * x1 + np.random.uniform(size=n)
        X = pd.DataFrame(np.array([x0, x1, x2, x3]).T, columns=["x0", "x1", "x2", "x3"])
        X_list.append(X)

    return X_list


def test_fit_success():
    X_list = create_data()

    reg = RandomForestRegressor(max_depth=4, random_state=0)
    model = MultiGroupRESIT(regressor=reg)
    model.fit(X_list)

    assert len(model.adjacency_matrices_) == 3
    assert len(model.causal_order_) == 4

    model.estimate_total_effect(X_list[0], 0, 0)
    model.get_error_independence_p_values(X_list[0])


def test_fit_invalid():
    X_list = create_data()

    reg = RandomForestRegressor(max_depth=4, random_state=0)

    # Not array data
    with pytest.raises(ValueError):
        model = MultiGroupRESIT(regressor=reg)
        model.fit(1)

    # Invalid number of groups
    with pytest.raises(ValueError):
        model = MultiGroupRESIT(regressor=reg)
        model.fit(X_list[:1])

    # Invalid number of groups
    with pytest.raises(ValueError):
        model = MultiGroupRESIT(regressor=reg)
        model.fit(
            [
                X_list[0][["x0", "x1", "x2"]],
                X_list[1][["x0", "x1"]],
                X_list[2][["x0", "x1", "x2"]],
            ]
        )

    # Invalid regressor
    with pytest.raises(ValueError):
        model = MultiGroupRESIT(regressor=None)
        model.fit(X_list)

    # Invalid regressor
    dummy_reg = lambda x: x
    with pytest.raises(ValueError):
        model = MultiGroupRESIT(regressor=dummy_reg)
        model.fit(X_list)

    # Invalid alpha
    with pytest.raises(ValueError):
        model = MultiGroupRESIT(regressor=reg, alpha=-1)
        model.fit(X_list)


def test_bootstrap_success():
    X_list = create_data()

    reg = RandomForestRegressor(max_depth=4, random_state=0)
    model = MultiGroupRESIT(regressor=reg)
    result_list = model.bootstrap(X_list, n_sampling=5)
    result = result_list[0]
    assert len(result.adjacency_matrices_) == 5
    assert len(result.total_effects_) == 5

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
    X_list = create_data()
    reg = RandomForestRegressor(max_depth=4, random_state=0)
    model = MultiGroupRESIT(regressor=reg)

    # Invalid argument: bootstrap(n_sampling=-1)
    with pytest.raises(ValueError):
        result_list = model.bootstrap(X_list, n_sampling=-1)

    # Invalid argument: bootstrap(n_sampling='3')
    with pytest.raises(ValueError):
        result_list = model.bootstrap(X_list, n_sampling="3")

    result_list = model.bootstrap(X_list, n_sampling=5)

    # Invalid argument: get_causal_direction_counts(n_directions=-1)
    with pytest.raises(ValueError):
        result_list[0].get_causal_direction_counts(n_directions=-1)

    # Invalid argument: get_causal_direction_counts(n_directions='3')
    with pytest.raises(ValueError):
        result_list[0].get_causal_direction_counts(n_directions="3")

    # Invalid argument: get_causal_direction_counts(min_causal_effect=-1.0)
    with pytest.raises(ValueError):
        result_list[0].get_causal_direction_counts(min_causal_effect=-1.0)

    # Invalid argument: get_directed_acyclic_graph_counts(n_dags=-1)
    with pytest.raises(ValueError):
        result_list[0].get_directed_acyclic_graph_counts(n_dags=-1)

    # Invalid argument: get_directed_acyclic_graph_counts(n_dags='3')
    with pytest.raises(ValueError):
        result_list[0].get_directed_acyclic_graph_counts(n_dags="3")

    # Invalid argument: get_directed_acyclic_graph_counts(min_causal_effect=-1.0)
    with pytest.raises(ValueError):
        result_list[0].get_directed_acyclic_graph_counts(min_causal_effect=-1.0)

    # Invalid argument: get_probabilities(min_causal_effect=-1.0)
    with pytest.raises(ValueError):
        result_list[0].get_probabilities(min_causal_effect=-1.0)

    # Invalid argument: get_total_causal_effects(min_causal_effect=-1.0)
    with pytest.raises(ValueError):
        result_list[0].get_total_causal_effects(min_causal_effect=-1.0)

    # Invalid argument: get_paths(min_causal_effect=-1.0)
    with pytest.raises(ValueError):
        result_list[0].get_paths(0, 1, min_causal_effect=-1.0)


def test_prior_knowledge_success():
    X_list = create_data()

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
    model = MultiGroupRESIT(regressor=reg, prior_knowledge=pk)
    model.fit(X_list)
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
    model = MultiGroupRESIT(regressor=reg, prior_knowledge=pk)
    model.fit(X_list)

    # prior knowledge: x0 is sink
    pk = np.array(
        [
            [0, -1, -1, -1],
            [0, 0, -1, -1],
            [0, -1, 0, -1],
            [0, -1, -1, 0],
        ]
    )
    model = MultiGroupRESIT(regressor=reg, prior_knowledge=pk)
    model.fit(X_list)

    # prior knowledge: x2-->x3 has path
    pk = np.array(
        [
            [0, -1, -1, -1],
            [-1, 0, -1, -1],
            [-1, -1, 0, 0],
            [-1, -1, 1, 0],
        ]
    )
    model = MultiGroupRESIT(regressor=reg, prior_knowledge=pk)
    model.fit(X_list)

    # prior knowledge: x1-->x3 does not have path
    pk = np.array(
        [
            [0, -1, -1, -1],
            [-1, 0, -1, -1],
            [-1, -1, 0, -1],
            [-1, 0, -1, 0],
        ]
    )
    model = MultiGroupRESIT(regressor=reg, prior_knowledge=pk)
    model.fit(X_list)
    co = model.causal_order_


def test_prior_knowledge_invalid():
    # causal direction: x0 --> x1 --> x3
    X_list = create_data(1000)

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
    with pytest.raises(ValueError):
        model = MultiGroupRESIT(regressor=reg, prior_knowledge=pk)
        model.fit(X_list)

    # prior knowledge: inconsistent
    pk = np.array(
        [
            [0, 1, -1, -1],
            [1, 0, -1, -1],
            [-1, -1, 0, -1],
            [-1, -1, -1, 0],
        ]
    )
    with pytest.raises(ValueError):
        model = MultiGroupRESIT(regressor=reg, prior_knowledge=pk)
        model.fit(X_list)
