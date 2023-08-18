import os
import pytest

import numpy as np
import pandas as pd

from lingam import RESIT
from lingam.utils import visualize_nonlinear_causal_effect

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import LinearSVC


@pytest.fixture
def test_data_adj():
    np.random.seed(0)

    X = _generate_data()

    reg = RandomForestRegressor()
    model = RESIT(regressor=reg, alpha=1)
    model.fit(X)

    adj_mat = model.adjacency_matrix_
    cause_name = "X2"
    effect_name = "X3"
    return X, adj_mat, cause_name, effect_name

@pytest.fixture
def test_data_bs():
    np.random.seed(0)

    X = _generate_data()

    reg = RandomForestRegressor()
    model = RESIT(regressor=reg, alpha=1)

    bs_result = model.bootstrap(X, n_sampling=3)
    cause_name = "X2"
    effect_name = "X3"
    return X, bs_result, cause_name, effect_name

def _generate_data(n_samples=100):
    def N1(size):
        return np.random.uniform(size=size) - 0.5

    X = np.zeros([n_samples, 5])
    X[:, 0] = N1(n_samples)
    X[:, 1] = 3 * (X[:, 0] + 0.25) * (X[:, 0] - 0.25) + N1(n_samples)
    X[:, 2] = -0.75 * (X[:, 0] - 1) * (X[:, 0] - 2) + 1.5 * X[:, 1] + N1(n_samples)
    X[:, 3] = 5 * (X[:, 1] + 0.4) * (X[:, 1] - 0.1) * (X[:, 1] - 0.5) + 1 * np.log(5 * X[:, 2] + 20) + N1(n_samples)
    X[:, 4] = -0.8 * (X[:, 3] - 1.5) * (X[:, 3] - 3.5) + N1(n_samples)

    X = pd.DataFrame(X, columns=[f"X{i}" for i in range(5)])
    return X

def test_exec_sucess_adj(test_data_adj):
    X, adj_mat, cause_name, effect_name = test_data_adj

    # continuous
    fig = visualize_nonlinear_causal_effect(
        X,
        adj_mat,
        RandomForestRegressor(),
        cause_name,
        effect_name
    )
    assert len(fig.axes) == 1

    # discrete
    X2 = _to_discrete(X, effect_name)
    fig = visualize_nonlinear_causal_effect(
        X2,
        adj_mat,
        RandomForestClassifier(),
        cause_name,
        effect_name
    )
    assert len(fig.axes) == 3

def test_exec_sucess_bs(test_data_bs):
    X, bs_result, cause_name, effect_name = test_data_bs

    # continuous
    fig = visualize_nonlinear_causal_effect(
        X,
        bs_result,
        RandomForestRegressor(),
        cause_name,
        effect_name
    )
    assert len(fig.axes) == 1

    # discrete
    X2 = _to_discrete(X, effect_name)
    fig = visualize_nonlinear_causal_effect(
        X2,
        bs_result,
        RandomForestClassifier(),
        cause_name,
        effect_name
    )
    assert len(fig.axes) == 3

def _to_discrete(X, name, q=(33, 66)):
    X = X.copy()
    delims = np.percentile(X[name], q=q)
    X[name] = X[name].map(lambda x: sum([delim <= x for delim in delims])) 
    X[name] = pd.Categorical(X[name])
    return X

def test_exec_exceptioin_args_check(test_data_adj):
    X, adj_mat, cause_name, effect_name = test_data_adj
    X2 = _to_discrete(X, effect_name)

    # X
    try:
        fig = visualize_nonlinear_causal_effect(
            np.array(X),
            adj_mat,
            RandomForestRegressor(),
            cause_name,
            effect_name
        )
    except TypeError:
        pass
    else:
        raise AssertionError

    # cd_result
    try:
        fig = visualize_nonlinear_causal_effect(
            X,
            adj_mat.reshape(-1, 1),
            RandomForestRegressor(),
            cause_name,
            effect_name
        )
    except TypeError:
        pass
    else:
        raise AssertionError

    try:
        fig = visualize_nonlinear_causal_effect(
            X,
            adj_mat.astype(str),
            RandomForestRegressor(),
            cause_name,
            effect_name
        )
    except ValueError:
        pass
    else:
        raise AssertionError

    # cause_name
    try:
        fig = visualize_nonlinear_causal_effect(
            X,
            adj_mat,
            RandomForestRegressor(),
            "cause_name",
            effect_name
        )
    except ValueError:
        pass
    else:
        raise AssertionError

    # effect_ame
    try:
        fig = visualize_nonlinear_causal_effect(
            X,
            adj_mat,
            RandomForestRegressor(),
            cause_name,
            "effect_name"
        )
    except ValueError:
        pass
    else:
        raise AssertionError

    # estmator
    try:
        fig = visualize_nonlinear_causal_effect(
            X2,
            adj_mat,
            RandomForestRegressor(),
            cause_name,
            effect_name
        )
    except TypeError:
        pass
    else:
        raise AssertionError

    try:
        fig = visualize_nonlinear_causal_effect(
            X,
            adj_mat,
            RandomForestClassifier(),
            cause_name,
            effect_name
        )
    except TypeError:
        pass
    else:
        raise AssertionError

    try:
        fig = visualize_nonlinear_causal_effect(
            X2,
            adj_mat,
            LinearSVC(),
            cause_name,
            effect_name
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError

    # cause_positions
    try:
        fig = visualize_nonlinear_causal_effect(
            X,
            adj_mat,
            RandomForestRegressor(),
            cause_name,
            effect_name,
            cause_positions=[1, 2, 3, None]
        )
    except TypeError:
        pass
    else:
        raise AssertionError

    # percentile
    try:
        fig = visualize_nonlinear_causal_effect(
            X,
            adj_mat,
            RandomForestRegressor(),
            cause_name,
            effect_name,
            percentile=(0, 30, 60, 100)
        )
    except TypeError:
        pass
    else:
        raise AssertionError

    try:
        fig = visualize_nonlinear_causal_effect(
            X,
            adj_mat,
            RandomForestRegressor(),
            cause_name,
            effect_name,
            percentile=(0, 50, 101)
        )
    except TypeError:
        pass
    else:
        raise AssertionError

    # fig
    try:
        fig = visualize_nonlinear_causal_effect(
            X,
            adj_mat,
            RandomForestRegressor(),
            cause_name,
            effect_name,
            fig = 0
        )
    except TypeError:
        pass
    else:
        raise AssertionError

