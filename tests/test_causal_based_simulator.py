import pytest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from scipy.special import expit

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lingam.causal_based_simulator import CausalBasedSimulator
from lingam.causal_based_simulator import CBSILiNGAM
from lingam.causal_based_simulator import CBSIUnobsCommonCauseLiNGAM
from lingam.causal_based_simulator import CBSITimeSeriesLiNGAM


@pytest.fixture
def init():
    return lambda : np.random.seed(0)


@pytest.fixture
def test_data():
    np.random.seed(0)

    N = 1000
    causal_graph = np.array([
        [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
        [3.0, 0.0, 2.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [8.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])

    def _is_correct_co(causal_order):
        truth = [{"3"}, {"0", "2"}, {"1", "4", "5"}]
        target = [
            {*causal_order[:1]},
            {*causal_order[1:3]},
            {*causal_order[3:]}
        ]
        return truth == target

    e = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=(len(causal_graph), N))
    X = np.linalg.pinv(np.eye(len(causal_graph)) - causal_graph) @ e
    X = X.T

    return X, causal_graph, _is_correct_co


@pytest.fixture
def test_data_unobs():
    np.random.seed(0)

    N = 1000
    causal_graph = np.array([
        [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
        [3.0, 0.0, 2.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [8.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    e = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=(len(causal_graph), N))
    X = np.linalg.pinv(np.eye(len(causal_graph)) - causal_graph) @ e
    X = X.T

    delete_index = 3
    causal_graph = np.delete(causal_graph, delete_index, axis=0)
    causal_graph = np.delete(causal_graph, delete_index, axis=1)
    X = np.delete(X, delete_index, axis=1)
    causal_graph[0, 2] = np.nan
    causal_graph[2, 0] = np.nan

    return X, causal_graph


@pytest.fixture
def test_data_ts():
    np.random.seed(0)

    N = 1000
    causal_graph = np.array([[
        [0, -0.12, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [-0.41, 0.01, 0, -0.02, 0],
        [0.04, -0.22, 0, 0, 0],
        [0.15, 0, -0.03, 0, 0],
    ], [
        [-0.32, 0, 0.12, 0.32, 0],
        [0, -0.35, -0.1, -0.46, 0.4],
        [0, 0, 0.37, 0, 0.46],
        [-0.38, -0.1, -0.24, 0, -0.13],
        [0, 0, 0, 0, 0],
    ]])

    def _is_correct_co(causal_order):
        truth = [
            {"0[t-1]", "1[t-1]", "2[t-1]", "3[t-1]", "4[t-1]"},
            {"1[t]"}, {"0[t]"}, {"3[t]"}, {"2[t]"}, {"4[t]"},
        ]
        target = [
            {*causal_order[:5]},
            {*causal_order[5:6]},
            {*causal_order[6:7]},
            {*causal_order[7:8]},
            {*causal_order[8:9]},
            {*causal_order[9:10]},
        ]
        return truth == target

    def _x_t(X, e, causal_graph):
        _, n_features, _ = causal_graph.shape
        term = np.linalg.pinv(np.eye(n_features) - causal_graph[0])
        term2 = np.hstack(causal_graph[1:]) @ np.hstack(X[:, ::-1][:, :len(causal_graph) - 1]).reshape(n_features, 1) + e
        return term @ term2

    n_lags = len(causal_graph) - 1
    size = N + n_lags

    errors = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=(causal_graph.shape[1], size))

    # initial data
    X = errors[:, :n_lags]

    for t in range(n_lags, size):
        new_data = _x_t(X[:, :t], errors[:, [t]], causal_graph)
        X = np.append(X, new_data, axis=1)
    X = X[:, n_lags:].T

    return X, causal_graph, _is_correct_co


@pytest.fixture
def test_data_discrete():
    np.random.seed(0)

    N = 1000
    causal_graph = np.array([
        [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
        [3.0, 0.0, 2.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [8.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])
    discrete_indices = [0, 1]
    causal_order = [3, 0, 2, 1, 4, 5]

    def _is_correct_co(causal_order):
        truth = [{"3"}, {"0", "2"}, {"1", "4", "5"}]
        target = [
            {*causal_order[:1]},
            {*causal_order[1:3]},
            {*causal_order[3:]}
        ]
        return truth == target

    def e():
        return np.random.uniform(-np.sqrt(3), np.sqrt(3), size=N)

    X = np.zeros((len(causal_graph), N))

    for co in causal_order:
        coefs = causal_graph[co]
        is_exog = np.all(np.isclose(coefs, 0))
        if is_exog:
            X[co] = e()
            continue

        X[co] = coefs @ X + e()
        if co in discrete_indices:
            X[co] = (expit(X[co]) > np.random.uniform(size=N)).astype(int)

    X = X.T

    is_discrete = np.array([False for _ in range(len(causal_graph))])
    is_discrete[discrete_indices] = True

    return X, causal_graph, _is_correct_co, is_discrete


def test_cbs_success(test_data, test_data_unobs, test_data_ts, test_data_discrete):
    models = {"0": LinearRegression()}
    models_d = {"0": LogisticRegression()}
    models_ts = {"0[t]": LinearRegression()}
    changing_models = {"0": {"parent_names": []}}
    changing_models_ts = {"0[t]": {"parent_names": []}}
    changing_exog = {"2": np.random.uniform(-10, 10, size=1000)}
    changing_exog_ts = {"2[t]": np.random.uniform(-10, 10, size=999)}

    sim = CausalBasedSimulator()

    # normal data
    X, causal_graph, _ = test_data
    sim.train(X, causal_graph)
    sim.train(X, causal_graph, models=models)
    sim.run()
    sim.run(changing_models=changing_models, changing_exog=changing_exog)

    # unobserved
    X, causal_graph = test_data_unobs
    sim.train(X, causal_graph, cd_algo_name="BottomUpParceLiNGAM")
    sim.train(X, causal_graph, cd_algo_name="BottomUpParceLiNGAM", models=models)
    sim.run()
    sim.run(changing_models=changing_models, changing_exog=changing_exog)

    # time series
    X, causal_graph, _ = test_data_ts
    sim.train(X, causal_graph, cd_algo_name="VARLiNGAM")
    sim.train(X, causal_graph, cd_algo_name="VARLiNGAM", models=models_ts)
    sim.run()
    sim.run(changing_models=changing_models_ts, changing_exog=changing_exog_ts)

    # discrete
    X, causal_graph, _, is_discrete = test_data_discrete
    sim.train(X, causal_graph, is_discrete=is_discrete)
    sim.train(X, causal_graph, is_discrete=is_discrete, models=models_d)
    sim.run()
    sim.run(changing_models=changing_models, changing_exog=changing_exog)


def test_cbs_exception(test_data):
    X, causal_graph, _ = test_data

    sim = CausalBasedSimulator()

    # cd_algo_name
    try:
        sim.train(X, causal_graph, cd_algo_name=1234)
    except Exception:
        pass
    else:
        raise AssertionError

    try:
        sim.train(X, causal_graph, cd_algo_name="UnknownAlgoName")
    except Exception:
        pass
    else:
        raise AssertionError

    # changing_models makes causal_graph cyclic
    changing_models = {
        "x3": {"parent_names" : ["x0"]}
    }
    sim.train(X, causal_graph)
    try:
        sim.run(changing_models=changing_models)
    except Exception:
        pass
    else:
        raise AssertionError


def test_cbsi_lingam_success(test_data, test_data_discrete):
    X, causal_graph, is_correct_co = test_data

    # constructor

    # X
    CBSILiNGAM(X, causal_graph)

    # X is pandas.DataFrame
    X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    CBSILiNGAM(X_df, causal_graph)

    # properties
    impl = CBSILiNGAM(X, causal_graph)
    assert is_correct_co(impl.causal_order_)
    assert impl.endog_names_ == [f"{i}" for i in range(len(causal_graph))]
    assert impl.discrete_endog_names_ == []
    assert impl.exog_length_ == len(X)

    # get_paraent_names
    parent = impl.get_parent_names("1")
    assert parent == ["0", "2"]
    parent = impl.get_parent_names("3")
    assert parent == []

    # get_data
    data = impl.get_data("0")
    assert np.all(data == X[:, [0]])

    # get_causal_order
    causal_order = impl.get_causal_order()
    assert is_correct_co(causal_order)


def test_cbsi_lingam_success2(test_data_discrete):
    X, causal_graph, is_correct_co, is_discrete = test_data_discrete

    # constructor

    # X
    CBSILiNGAM(X, causal_graph, is_discrete=is_discrete)

    # X is pandas.DataFrame
    X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    CBSILiNGAM(X_df, causal_graph, is_discrete)

    # properties

    impl = CBSILiNGAM(X, causal_graph, is_discrete=is_discrete)
    assert is_correct_co(impl.causal_order_)
    assert impl.endog_names_ == [f"{i}" for i in range(len(causal_graph))]
    assert len(impl.discrete_endog_names_) == 2

    # get_paraent_names
    parent = impl.get_parent_names("1")
    assert parent == ["0", "2"]
    parent = impl.get_parent_names("3")
    assert parent == []

    # get_data
    data = impl.get_data("0")
    assert np.all(data == X[:, [0]])

    # get_causal_order
    causal_order = impl.get_causal_order()
    assert is_correct_co(causal_order)


def test_cbsi_lingam_exception(test_data):
    X, causal_graph, _ = test_data

    # X: wrong dim
    try:
        CBSILiNGAM(X.reshape(1, *X.shape), causal_graph)
    except Exception:
        pass
    else:
        raise AssertionError

    # X: nan
    X_ = X.copy()
    X_[0, 0] = np.nan
    try:
        CBSILiNGAM(X_, causal_graph)
    except Exception:
        pass
    else:
        raise AssertionError

    # causal_graph: wrong shape
    try:
        CBSILiNGAM(X, np.concatenate([causal_graph] * 2))
    except Exception:
        pass
    else:
        raise AssertionError

    # causal_graph: wrong dim
    try:
        CBSILiNGAM(X, causal_graph.reshape(1, *causal_graph.shape))
    except Exception:
        pass
    else:
        raise AssertionError

    # causal_graph: cyclic
    causal_graph_cyclic = causal_graph.copy()
    causal_graph_cyclic[3, 5] = 1
    try:
        CBSILiNGAM(X, causal_graph_cyclic)
    except Exception:
        pass
    else:
        raise AssertionError

    # is_discrete: wrong length
    is_discrete = [False for i in range(len(causal_graph) + 1)]
    try:
        CBSILiNGAM(X, causal_graph, is_discrete=is_discrete)
    except Exception:
        pass
    else:
        raise AssertionError


def test_cbsi_unobs_success(test_data_unobs):
    X, causal_graph = test_data_unobs

    # X
    CBSIUnobsCommonCauseLiNGAM(X, causal_graph)

    # X is pandas.DataFrame
    X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    CBSIUnobsCommonCauseLiNGAM(X_df, causal_graph)


def test_cbsi_ts_success(test_data_ts):
    X, causal_graph, is_correct_co = test_data_ts

    # constructor

    # X
    CBSITimeSeriesLiNGAM(X, causal_graph)

    # X is pandas.DataFrame
    X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    CBSITimeSeriesLiNGAM(X_df, causal_graph)

    # properties

    impl = CBSITimeSeriesLiNGAM(X, causal_graph)
    assert impl.endog_names_ == [*[f"{i}[t]" for i in range(X.shape[1])], *[f"{i}[t-1]" for i in range(X.shape[1])]]

    # get_paraent_names
    parent = impl.get_parent_names("1[t]")
    assert parent == ["1[t-1]", "2[t-1]", "3[t-1]", "4[t-1]"]
    parent = impl.get_parent_names("1[t-1]")
    assert parent == []

    # get_data
    data = impl.get_data("1[t]")
    n_lags = len(causal_graph) - 1
    assert data.shape == (len(X) - n_lags, 1)

    # get_causal_order
    causal_order = impl.get_causal_order()
    assert is_correct_co(causal_order)


def test_cbsi_ts_exception(test_data_ts):
    X, causal_graph, is_correct_co = test_data_ts

    # X: wrong dim
    try:
        CBSITimeSeriesLiNGAM(X.reshape(1, *X.shape), causal_graph)
    except Exception:
        pass
    else:
        raise AssertionError

    # X: wrong n_features
    X_ = np.concatenate([X, X], axis=1)
    try:
        CBSITimeSeriesLiNGAM(X_, causal_graph)
    except Exception:
        pass
    else:
        raise AssertionError

    # causal_graph: wrong dim
    try:
        CBSITimeSeriesLiNGAM(X, causal_graph.reshape(1, *causal_graph.shape))
    except Exception:
        pass
    else:
        raise AssertionError

    # causal_graph: wrong length
    causal_graph_ = np.concatenate([causal_graph] * 2, axis=1)
    try:
        CBSITimeSeriesLiNGAM(X, causal_graph_)
    except Exception:
        pass
    else:
        raise AssertionError
