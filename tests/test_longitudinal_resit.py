import os

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from lingam.longitudinal_resit import LongitudinalRESIT


def generate_data_common(n_samples=300, T=6, seed=0):

    B = np.zeros((5, 5))
    B[2, 0] = 1  # x0→x2
    B[3, 1] = 1  # x1→x3
    B[4, 2] = 1  # x2→x4
    B[4, 3] = 1  # x3→x4

    A = np.zeros((1, 5, 5))
    A[0, 0, 0] = 1  # x0(t-1)→x0(t)
    A[0, 1, 1] = 1  # x1(t-1)→x1(t)
    A[0, 4, 3] = 1  # x3(t-1)→x4(t)

    rng = np.random.default_rng(seed)

    def noise(size, scale=0.3):
        return rng.uniform(-scale, scale, size)

    X_list = []

    for t in range(T):
        if t == 0:
            x0 = rng.uniform(-1, 1, n_samples)
            x1 = rng.uniform(-1, 1, n_samples)
            x2 = x0**2 + noise(n_samples)
            x3 = np.sin(x1) + noise(n_samples)
            x4 = np.tanh(x2) + 0.5 * x3**2 + noise(n_samples)
        else:
            X_prev = X_list[t - 1]
            x0 = 0.6 * X_prev[:, 0] + noise(n_samples, 0.2)
            x1 = 0.5 * X_prev[:, 1] + noise(n_samples, 0.2)
            x2 = x0**2 + noise(n_samples)
            x3 = np.sin(x1) + noise(n_samples)
            x4 = np.tanh(x2) + 0.5 * x3**2 + 0.4 * X_prev[:, 3] + noise(n_samples)

        X_list.append(np.column_stack([x0, x1, x2, x3, x4]))

    return [B], [A], X_list


def generate_data_separate(n_samples=300, T=6, seed=0):
    B_list = []
    for t in range(T):
        if t == 0:
            B_list.append(np.full((5, 5), np.nan))
            continue
        B = np.zeros((5, 5))
        if t <= 2:
            B[2, 0] = 1  # x0→x2
            B[3, 1] = 1  # x1→x3
            B[4, 2] = 1  # x2→x4
            B[4, 3] = 1  # x3→x4
        else:
            B[2, 0] = 1  # x0→x2
            B[4, 2] = 1  # x2→x4
            B[4, 1] = 1  # x1→x4
        B_list.append(B)

    A_list = []
    for t in range(T):
        if t == 0:
            A_list.append(np.full((1, 5, 5), np.nan))
            continue
        A = np.zeros((1, 5, 5))
        A[0, 0, 0] = 1  # x0(t-1)→x0(t)
        A[0, 1, 1] = 1  # x1(t-1)→x1(t)
        if t >= 3:
            A[0, 4, 3] = 1  # x3(t-1)→x4(t)
        A_list.append(A)

    rng = np.random.default_rng(seed)

    def noise(size, scale=0.3):
        return rng.uniform(-scale, scale, size)

    X_list = []

    for t in range(T):
        if t == 0:
            x0 = rng.uniform(-1, 1, n_samples)
            x1 = rng.uniform(-1, 1, n_samples)
            x2 = x0**2 + noise(n_samples)
            x3 = np.sin(x1) + noise(n_samples)
            x4 = np.tanh(x2) + 0.5 * x3**2 + noise(n_samples)

        elif t <= 2:
            X_prev = X_list[t - 1]
            x0 = 0.6 * X_prev[:, 0] + noise(n_samples, 0.2)
            x1 = 0.5 * X_prev[:, 1] + noise(n_samples, 0.2)
            x2 = x0**2 + noise(n_samples)
            x3 = np.sin(x1) + noise(n_samples)
            x4 = np.tanh(x2) + 0.5 * x3**2 + noise(n_samples)

        else:
            X_prev = X_list[t - 1]
            x0 = 0.6 * X_prev[:, 0] + noise(n_samples, 0.2)
            x1 = 0.5 * X_prev[:, 1] + noise(n_samples, 0.2)
            x2 = x0**2 + noise(n_samples)
            x3 = noise(n_samples)
            x4 = np.tanh(x2) + 0.6 * x1**2 + 0.4 * X_prev[:, 3] + noise(n_samples)

        X_list.append(np.column_stack([x0, x1, x2, x3, x4]))

    return B_list, A_list, X_list


def test_fit_success():

    # Common causal graph across time points
    _, _, X_list = generate_data_common(n_samples=300, T=6, seed=0)
    model = LongitudinalRESIT(
        regressor=RandomForestRegressor(n_estimators=100, random_state=0),
        n_lags=1,
        alpha=0.05,
        is_common_graph=True,
    )
    model.fit(X_list)

    # Separate causal graph across time points
    _, _, X_list = generate_data_separate(n_samples=300, T=6, seed=0)
    model = LongitudinalRESIT(
        regressor=RandomForestRegressor(n_estimators=100, random_state=0),
        n_lags=1,
        alpha=0.05,
        is_common_graph=False,
    )
    model.fit(X_list)

    # Prior knowledge for common causal graph across time points
    _, _, X_list = generate_data_common(n_samples=300, T=6, seed=0)
    n_vars = X_list[0].shape[1]
    pk = -np.ones((n_vars, n_vars))
    pk[2, 0] = pk[3, 1] = pk[4, 3] = 1
    pk[2, 1] = 0
    pkl = -np.ones((1, n_vars, n_vars))
    pkl[0, 0, 0] = pkl[0, 1, 1] = pkl[0, 4, 3] = 1
    pkl[0, 2, 3] = pkl[0, 2, 4] = pkl[0, 3, 0] = 0
    model = LongitudinalRESIT(
        regressor=RandomForestRegressor(n_estimators=100, random_state=0),
        n_lags=1,
        alpha=0.05,
        prior_knowledge=pk,
        prior_knowledge_lag=pkl,
        is_common_graph=True,
    )
    model.fit(X_list)

    # Prior knowledge for separate causal graph across time points
    _, _, X_list = generate_data_separate(n_samples=300, T=6, seed=0)
    n_vars = X_list[0].shape[1]
    pk = -np.ones((n_vars, n_vars))
    pk[2, 0] = pk[3, 1] = pk[4, 3] = 1
    pk[2, 1] = 0
    pkl = -np.ones((1, n_vars, n_vars))
    pkl[0, 0, 0] = pkl[0, 1, 1] = pkl[0, 4, 3] = 1
    pkl[0, 2, 3] = pkl[0, 2, 4] = pkl[0, 3, 0] = 0
    model = LongitudinalRESIT(
        regressor=RandomForestRegressor(n_estimators=100, random_state=0),
        n_lags=1,
        alpha=0.05,
        prior_knowledge=pk,
        prior_knowledge_lag=pkl,
        is_common_graph=False,
    )
    model.fit(X_list)


def test_fit_invalid_data():

    reg = RandomForestRegressor(n_estimators=100, random_state=0)
    model = LongitudinalRESIT(
        regressor=reg,
        n_lags=2,
        alpha=0.05,
        is_common_graph=True,
    )

    # Invalid data: empty list
    try:
        model.causal_order_
    except ValueError as e:
        pass
    else:
        raise AssertionError

    # Invalid data: empty list
    try:
        model.B_list_
    except ValueError as e:
        pass
    else:
        raise AssertionError

    # Invalid data: empty list
    try:
        model.A_list_
    except ValueError as e:
        pass
    else:
        raise AssertionError

    # Invalid data: empty list
    try:
        model.adjacency_matrices_
    except ValueError as e:
        pass
    else:
        raise AssertionError

    # Invalid data: not a list of arrays
    X_list = np.random.rand(300, 5)
    try:
        model.fit(X_list)
    except ValueError as e:
        pass
    else:
        raise AssertionError

    # Invalid data: arrays with different number of samples
    X_list = [np.random.rand(300, 5), np.random.rand(200, 5), np.random.rand(200, 5)]
    try:
        model.fit(X_list)
    except ValueError as e:
        pass
    else:
        raise AssertionError

    # Invalid data: Lag variables are only available for some time points.
    X_list = [np.random.rand(300, 5), np.random.rand(200, 5)]
    try:
        model.fit(X_list)
    except ValueError as e:
        pass
    else:
        raise AssertionError

    # Invalid data: prior_knowledge must be a single matrix of shape
    model = LongitudinalRESIT(
        regressor=reg,
        n_lags=1,
        alpha=0.05,
        prior_knowledge=1,
        prior_knowledge_lag=None,
        is_common_graph=True,
    )
    X_list = [np.random.rand(300, 5), np.random.rand(300, 5)]
    try:
        model.fit(X_list)
    except ValueError as e:
        pass
    else:
        raise AssertionError

    # Invalid data:prior_knowledge shape must be ({n_vars}, {n_vars})
    model = LongitudinalRESIT(
        regressor=reg,
        n_lags=1,
        alpha=0.05,
        prior_knowledge=np.ones((5, 4)),
        prior_knowledge_lag=None,
        is_common_graph=True,
    )
    X_list = [np.random.rand(300, 5), np.random.rand(300, 5)]
    try:
        model.fit(X_list)
    except ValueError as e:
        pass
    else:
        raise AssertionError

    # Invalid data: prior_knowledge_lag must be a single matrix of shape
    model = LongitudinalRESIT(
        regressor=reg,
        n_lags=1,
        alpha=0.05,
        prior_knowledge=None,
        prior_knowledge_lag=1,
        is_common_graph=True,
    )
    X_list = [np.random.rand(300, 5), np.random.rand(300, 5)]
    try:
        model.fit(X_list)
    except ValueError as e:
        pass
    else:
        raise AssertionError

    # Invalid data: prior_knowledge_lag shape must be ({n_vars}, {n_vars})
    model = LongitudinalRESIT(
        regressor=reg,
        n_lags=1,
        alpha=0.05,
        prior_knowledge=None,
        prior_knowledge_lag=np.ones((5, 4)),
        is_common_graph=True,
    )
    X_list = [np.random.rand(300, 5), np.random.rand(300, 5)]
    try:
        model.fit(X_list)
    except ValueError as e:
        pass
    else:
        raise AssertionError

    # Invalid data: The prior knowledge contains inconsistencies
    model = LongitudinalRESIT(
        regressor=reg,
        n_lags=1,
        alpha=0.05,
        prior_knowledge=np.array([[0, 1], [1, 0]]),
        prior_knowledge_lag=None,
        is_common_graph=True,
    )
    X_list = [np.random.rand(300, 2), np.random.rand(300, 2)]
    try:
        model.fit(X_list)
    except ValueError as e:
        pass
    else:
        raise AssertionError

    # Invalid data: Specify regression model in 'regressor'
    X_list = [np.random.rand(300, 5), np.random.rand(300, 5)]
    try:
        model = LongitudinalRESIT(
            regressor=None,
            n_lags=1,
            alpha=0.05,
            is_common_graph=True,
        )
    except ValueError as e:
        pass
    else:
        raise AssertionError

    # Invalid data: 'regressor' has no fit or predict method.
    class InvalidRegressor:
        def fit(self, X, y):
            pass

    try:
        model = LongitudinalRESIT(
            regressor=InvalidRegressor(),
            n_lags=1,
            alpha=0.05,
            is_common_graph=True,
        )
    except ValueError as e:
        pass
    else:
        raise AssertionError

    # Invalid data: alpha must be a float greater than 0.
    try:
        model = LongitudinalRESIT(
            regressor=RandomForestRegressor(n_estimators=100, random_state=0),
            n_lags=1,
            alpha=-0.1,
            is_common_graph=True,
        )
    except ValueError as e:
        pass
    else:
        raise AssertionError

    # Invalid data: n_lags must be an integer greater than or equal to 1.
    try:
        model = LongitudinalRESIT(
            regressor=RandomForestRegressor(n_estimators=100, random_state=0),
            n_lags=0,
            alpha=0.05,
            is_common_graph=True,
        )
    except ValueError as e:
        pass
    else:
        raise AssertionError


def test_bootstrap_success():

    # Common causal graph across time points
    _, _, X_list = generate_data_common(n_samples=300, T=6, seed=0)
    model = LongitudinalRESIT(
        regressor=RandomForestRegressor(n_estimators=100, random_state=0),
        n_lags=1,
        alpha=0.05,
        is_common_graph=True,
    )
    result = model.bootstrap(X_list, n_sampling=2)

    # No argument
    cdc = result.get_causal_direction_counts()

    # n_directions=2
    cdc = result.get_causal_direction_counts(n_directions=2)

    # No argument
    dagc = result.get_directed_acyclic_graph_counts()

    # n_dags=2
    dagc = result.get_directed_acyclic_graph_counts(n_dags=2)

    # get_probabilities
    probs = result.get_probabilities()

    # get_paths
    result.get_paths(0, 1, 1, 3)

    # Separate causal graph across time points
    _, _, X_list = generate_data_separate(n_samples=300, T=6, seed=0)
    model = LongitudinalRESIT(
        regressor=RandomForestRegressor(n_estimators=100, random_state=0),
        n_lags=1,
        alpha=0.05,
        is_common_graph=False,
    )
    result = model.bootstrap(X_list, n_sampling=2)

    # No argument
    cdc = result.get_causal_direction_counts()

    # n_directions=2
    cdc = result.get_causal_direction_counts(n_directions=2)

    # No argument
    dagc = result.get_directed_acyclic_graph_counts()

    # n_dags=2
    dagc = result.get_directed_acyclic_graph_counts(n_dags=2)

    # get_probabilities
    probs = result.get_probabilities()

    # get_paths
    result.get_paths(0, 1, 1, 3)


def test_bootstrap_invalid_data():

    model = LongitudinalRESIT(
        regressor=RandomForestRegressor(n_estimators=100, random_state=0),
        n_lags=2,
        alpha=0.05,
        is_common_graph=True,
    )

    # Invalid data: not a list of arrays
    X_list = np.random.rand(300, 5)
    try:
        model.bootstrap(X_list, n_sampling=2)
    except ValueError as e:
        pass
    else:
        raise AssertionError

    # Invalid data: Lag variables are only available for some time points.
    X_list = [np.random.rand(300, 5), np.random.rand(200, 5)]
    try:
        model.bootstrap(X_list, n_sampling=2)
    except ValueError as e:
        pass
    else:
        raise AssertionError

    X_list = [np.random.rand(300, 5), np.random.rand(300, 5)]
    model = LongitudinalRESIT(
        regressor=RandomForestRegressor(n_estimators=100, random_state=0),
        n_lags=2,
        alpha=0.05,
        is_common_graph=True,
    )
    result = model.bootstrap(X_list, n_sampling=2)

    # Invalid data: n_directions must be an integer greater than 0
    try:
        result.get_causal_direction_counts(n_directions=0)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid data: n_directions must be an integer greater than 0
    try:
        result.get_causal_direction_counts(n_directions="a")
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid data: n_dags must be an integer greater than 0
    try:
        result.get_directed_acyclic_graph_counts(n_dags=0)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid data: n_dags must be an integer greater than 0
    try:
        result.get_directed_acyclic_graph_counts(n_dags="a")
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid data: to_t should be greater than or equal to from_t.
    try:
        result.get_paths(1, 0, 1, 0)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: The same variable is specified for from and to.
    try:
        result.get_paths(0, 0, 1, 1)
    except ValueError:
        pass
    else:
        raise AssertionError
