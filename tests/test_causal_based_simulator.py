import pickle
import pytest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy.special import expit

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lingam.causal_based_simulator import CausalBasedSimulator, TrainResult

DATA_DIR_PATH = os.path.dirname(__file__) + "/test_causal_based_simulator"

ENABLE_CAPTURE = False
if ENABLE_CAPTURE:
    if not os.path.isdir(DATA_DIR_PATH):
        os.mkdir(DATA_DIR_PATH)

@pytest.fixture
def init():
    return lambda :np.random.seed(0)

@pytest.fixture
def test_data():
    np.random.seed(0)

    N = 1000

    G = np.array([[0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                  [3.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [8.0, 0.0,-1.0, 0.0, 0.0, 0.0],
                  [4.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    X = {}
    X["x3"] = np.random.uniform(size=N)
    X["x2"] = G[2, 3] * X["x3"] + np.random.uniform(size=N)
    X["x0"] = G[0, 3] * X["x3"] + np.random.uniform(size=N)
    X["x4"] = G[4, 2] * X["x2"] + G[4, 0] * X["x0"] + np.random.uniform(size=N)
    X["x1"] = G[1, 2] * X["x2"] + G[1, 0] * X["x0"] + np.random.uniform(size=N)
    X["x5"] = G[5, 0] * X["x0"] + np.random.uniform(size=N)
    X = dict(sorted(X.items()))
    X = pd.DataFrame(X)

    return X, G

@pytest.fixture
def test_data2():
    np.random.seed(0)

    N = 1000

    G = np.array([[0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                  [3.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 1.5, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [8.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                  [4.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    
    X = {}
    
    X["x3"] = np.random.uniform(size=N)
    
    X["x2"] = np.empty(N).astype(str)
    for i in range(N):
        prob_a = expit(G[2, 3] * X["x3"][i] - 0.5)
        X["x2"][i] = np.random.choice(["a", "b"], p=[prob_a, 1 - prob_a])
    
    X["x0"] = G[0, 3] * X["x3"] + np.random.uniform(size=N)
    
    X["x4"] = np.empty(N)
    for i in range(N):
        if X["x2"][i] == "a":
            X["x4"][i] = 8.0 * X["x0"][i] + np.random.uniform()
        elif X["x2"][i] == "b":
            X["x4"][i] = 4.0 * X["x0"][i] + np.random.uniform()
    
    X["x1"] = np.empty(N)
    for i in range(N):
        if X["x2"][i] == "a":
            X["x1"][i] = 3.0 * X["x0"][i] + np.random.uniform()
        elif X["x2"][i] == "b":
            X["x1"][i] = 1.5 * X["x0"][i] + np.random.uniform()
    
    X["x5"] = G[5, 0] * X["x0"] + np.random.uniform(size=1000)
    
    X = dict(sorted(X.items()))
    X = pd.DataFrame(X)
    X["x2"] = X["x2"].astype("category")

    return X, G

@pytest.fixture
def test_data3():
    np.random.seed(0)

    G = np.array([[0.0, 0.0, 0.0],
                  [2.0, 0.0, 0.0],
                  [2.0, 0.0, 0.0]])

    N = 1000
    
    X = {}
    
    X["x0"] = np.random.choice(["a", "b"], p=[0.75, 0.25], size=N)
    
    X["x1"] = []
    for i in range(N):
        if X["x0"][i] == "a":
            X["x1"].append(np.random.uniform() * 2)
        elif X["x0"][i] == "b":
            X["x1"].append(np.random.uniform())
            
    X["x2"] = []
    for i in range(N):
        if X["x0"][i] == "a":
            X["x2"].append(np.random.choice(["c", "d"], p=[0.75, 0.25]))
        elif X["x0"][i] == "b":
            X["x2"].append(np.random.choice(["c", "d"], p=[0.25, 0.75]))
            
            
    X = pd.DataFrame(X)
    X["x0"] = X["x0"].astype("category")
    X["x2"] = X["x2"].astype("category")
    X = X[sorted(X.columns)]

    return X, G

def test_check_data(init, test_data, test_data2):
    init()

    X, causal_graph = test_data
    X2, causal_graph2 = test_data2

    # Normal
    sim = CausalBasedSimulator()
    try:
        X_, categorical_info = sim._check_data(X)
    except RuntimeError:
        raise AssertionError
    else:
        pass

    # Normal2: categorical variable
    sim = CausalBasedSimulator()
    try:
        X2_, categorical_info = sim._check_data(X2)
    except RuntimeError:
        raise AssertionError
    else:
        pass

    # Exception: wrog type
    X3 = np.ones((10, 5))
    sim = CausalBasedSimulator()
    try:
        X3_, categorical_info = sim._check_data(X3)
    except RuntimeError:
        pass
    else:
        raise AssertionError

    # Exception2: wrong type
    X4 = np.ones((10, 5)).tolist()
    sim = CausalBasedSimulator()
    try:
        X4_, categorical_info = sim._check_data(X4)
    except RuntimeError:
        pass
    else:
        raise AssertionError

def test_check_causal_graph(init, test_data):
    init()

    X, causal_graph = test_data

    # Normal
    sim = CausalBasedSimulator()
    try:
        causal_graph = sim._check_causal_graph(causal_graph, 6)
    except RuntimeError:
        raise AssertionError
    else:
        pass

    # Exception: wrong type
    sim = CausalBasedSimulator()
    try:
        causal_graph_ = sim._check_causal_graph("not array", 3)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Exception: wrong shape
    causal_graph2 = np.ones((3, 4))
    sim = CausalBasedSimulator()
    try:
        causal_graph2_ = sim._check_causal_graph(causal_graph2, 3)
    except RuntimeError:
        pass
    else:
        raise AssertionError

def test_check_models(init, test_data2):
    init()

    X, causal_graph = test_data2
    categorical_info = {"x2":["a", "b"]}

    # Normal
    sim = CausalBasedSimulator()
    models = {
        "x0": SVR(kernel="linear"),
        "x2": SVC(kernel="linear", probability=True),
        "x4": SVR(kernel="linear"),
    }
    try:
        sim._check_models(models, X.index, X.columns, categorical_info)
    except:
        raise AssertionError
    else:
        pass

    # Normal: Pipeline
    sim = CausalBasedSimulator()
    models = {
        "x0": Pipeline([
            ("preprocess", StandardScaler()),
            ("estimator", SVR(kernel="linear")),
        ]),
        "x2": SVC(kernel="linear", probability=True),
        "x4": SVR(kernel="linear"),
    }
    try:
        sim._check_models(models, X.index, X.columns, categorical_info)
    except:
        raise AssertionError
    else:
        pass

    # Normal：GridSearchCV
    sim = CausalBasedSimulator()
    models = {
        "x0": GridSearchCV(SVR(kernel="linear"), {"C": [0.1**i for i in range(3)]}),
        "x2": SVC(kernel="linear", probability=True),
        "x4": SVR(kernel="linear"),
    }
    try:
        sim._check_models(models, X.index, X.columns, categorical_info)
    except:
        raise AssertionError
    else:
        pass

    # Exception: wrong type
    sim = CausalBasedSimulator()
    models = (
        ("x0", SVR(kernel="linear")),
        ("x2", SVC(kernel="linear", probability=True)),
        ("x4", SVR(kernel="linear")),
    )
    try:
        sim._check_models(models, X.index, X.columns, categorical_info)
    except RuntimeError:
        pass
    else:
        raise AssertionError

    # Exception: wrong task type
    sim = CausalBasedSimulator()
    models = {
        "x0": SVC(kernel="linear", probability=True),
        "x2": SVC(kernel="linear", probability=True),
        "x4": SVR(kernel="linear"),
    }
    try:
        sim._check_models(models, X.index, X.columns, categorical_info)
    except RuntimeError:
        pass
    else:
        raise AssertionError


    # Exception: non existent variable name
    sim = CausalBasedSimulator()
    models = {
        "x0": SVR(kernel="linear"),
        "x2": SVC(kernel="linear", probability=True),
        "x6": SVR(kernel="linear"),
    }
    try:
        sim._check_models(models, X.index, X.columns, categorical_info)
    except RuntimeError:
        pass
    else:
        raise AssertionError

    # Exception: not instance
    sim = CausalBasedSimulator()
    models = {
        "x0": SVR(kernel="linear"),
        "x2": LogisticRegression,
        "x4": SVR(kernel="linear"),
    }
    try:
        sim._check_models(models, X.index, X.columns, categorical_info)
    except RuntimeError:
        pass
    else:
        raise AssertionError

    # Exception: Pipeline, wrong task
    sim = CausalBasedSimulator()
    models = {
        "x0": Pipeline([
            ("preprocess", StandardScaler()),
            ("estimator", SVC(kernel="linear", probability=True)),
        ]),
        "x2": SVC(kernel="linear", probability=True),
        "x4": SVR(kernel="linear"),
    }
    try:
        sim._check_models(models, X.index, X.columns, categorical_info)
    except:
        pass
    else:
        raise AssertionError

    # Exception：GridSearchCV, wrong task
    sim = CausalBasedSimulator()
    models = {
        "x0": GridSearchCV(SVC(kernel="linear", probability=True), {"C": [0.1**i for i in range(3)]}),
        "x2": SVC(kernel="linear", probability=True),
        "x4": SVR(kernel="linear"),
    }
    try:
        sim._check_models(models, X.index, X.columns, categorical_info)
    except:
        pass
    else:
        raise AssertionError

def test_changing_exog(init, test_data2):
    init()

    X, causal_graph = test_data2
    categorical_info = {"x2": ["a", "b"]}

    # Normal
    sim = CausalBasedSimulator()
    changing_exog = {
        "x0": np.random.uniform(size=X.shape[0]),
        "x1": np.random.uniform(size=X.shape[0]),
        "x3": np.random.uniform(size=X.shape[0]),
    }
    try:
        sim._check_changing_exog(changing_exog, X.index, X.columns, categorical_info)
    except RuntimeError:
        raise AssertionError
    else:
        pass

    # Exception: wrong type
    sim = CausalBasedSimulator()
    changing_exog = (
        ("x0", np.random.uniform(size=X.shape[0])),
        ("x1", np.random.uniform(size=X.shape[0])),
        ("x3", np.random.uniform(size=X.shape[0])),
    )
    try:
        sim._check_changing_exog(changing_exog, X.index, X.columns, categorical_info)
    except RuntimeError:
        pass
    else:
        raise AssertionError

    # Exception: non existent variable name
    sim = CausalBasedSimulator()
    changing_exog = {
        "x0": np.random.uniform(size=X.shape[0]),
        "x1": np.random.uniform(size=X.shape[0]),
        "x6": np.random.uniform(size=X.shape[0]),
    }
    try:
        sim._check_changing_exog(changing_exog, X.index, X.columns, categorical_info)
    except RuntimeError:
        pass
    else:
        raise AssertionError

    # Exception: can't specify to categorical variable
    sim = CausalBasedSimulator()
    changing_exog = {
        "x2": np.random.choice(["a", "b"], size=X.shape[0]),
    }
    try:
        sim._check_changing_exog(changing_exog, X.index, X.columns, categorical_info)
    except RuntimeError:
        pass
    else:
        raise AssertionError

    # Exception: wrong shape
    sim = CausalBasedSimulator()
    changing_exog = {
        "x0": np.random.uniform(size=X.shape[0]),
        "x1": np.random.uniform(size=X.shape[0] + 1),
        "x3": np.random.uniform(size=X.shape[0]),
    }
    try:
        sim._check_changing_exog(changing_exog, X.index, X.columns, categorical_info)
    except RuntimeError:
        pass
    else:
        raise AssertionError

def test_changing_models(init, test_data2):
    init()

    X, causal_graph = test_data2
    categorical_info = {"x2": ["a", "b"]}
    X["x2"] = X["x2"].apply(lambda x: categorical_info["x2"].index(x)).astype(int)
    train_results = {
        "x0": [
            TrainResult(condition=None, exp_columns=["x3"], model=None, predicted=None, residual=None),
        ],
        "x1": [
            TrainResult(condition={"x2": "a"}, exp_columns=["x0"], model=None, predicted=None, residual=None),
            TrainResult(condition={"x2": "b"}, exp_columns=["x0"], model=None, predicted=None, residual=None)
        ],
        "x2": [
            TrainResult(condition=None, exp_columns=["x3"], model=None, predicted=None, residual=None),
        ],
        "x3": [],
        "x4": [],
        "x5": [],
    }
    x0_model = LinearRegression()
    x0_model.fit(X[train_results["x0"][0].exp_columns].values, X["x0"] * 10)
    X_a = X[X["x2"] == categorical_info["x2"].index("a")]
    x1_model_a = LinearRegression()
    x1_model_a.fit(X_a[train_results["x1"][0].exp_columns].values, X_a["x1"] * 10)
    X_b = X[X["x2"] == categorical_info["x2"].index("b")]
    x1_model_b = LinearRegression()
    x1_model_b.fit(X_b[train_results["x1"][1].exp_columns].values, X_b["x1"] * 10)
    x2_model = LogisticRegression()
    x2_model.fit(X[train_results["x2"][0].exp_columns].values, X["x2"] * 10)
    x0_pipe = Pipeline([("estimator", LinearRegression())])
    x0_pipe.fit(X[train_results["x0"][0].exp_columns].values, X["x0"] * 10)
    x0_gs = GridSearchCV(SVR(kernel="linear"), {"C": [0.1**i for i in range(3)]})
    x0_gs.fit(X[train_results["x0"][0].exp_columns].values, X["x0"] * 10)

    # Normal
    sim = CausalBasedSimulator()
    changing_models = [
        {
            "name": "x0",
            "condition": None,
            "model": x0_model
        },
        {
            "name": "x1",
            "condition": {"x2": "a"},
            "model": x1_model_a
        },
        {
            "name": "x1",
            "condition": {"x2": "b"},
            "model": x1_model_b
        },
        {
            "name": "x2",
            "condition": None,
            "model": x2_model
        }
    ]
    try:
        sim._check_changing_models(changing_models, categorical_info, train_results)
    except RuntimeError:
        raise AssertionError
    else:
        pass

    # Normal: Pipeline
    sim = CausalBasedSimulator()
    changing_models = [
        {
            "name": "x0",
            "condition": None,
            "model": x0_pipe
        },
    ]
    try:
        sim._check_changing_models(changing_models, categorical_info, train_results)
    except RuntimeError:
        raise AssertionError
    else:
        pass

    # Normal: GridSearchCV
    sim = CausalBasedSimulator()
    changing_models = [
        {
            "name": "x0",
            "condition": None,
            "model": x0_gs
        },
    ]
    try:
        sim._check_changing_models(changing_models, categorical_info, train_results)
    except RuntimeError:
        raise AssertionError
    else:
        pass

    # Exception: type
    sim = CausalBasedSimulator()
    changing_models = {
        "x0": {
            "name": "x0",
            "condition": None,
            "model": x0_model
        },
    }
    try:
        sim._check_changing_models(changing_models, categorical_info, train_results)
    except RuntimeError:
        pass
    else:
        raise AssertionError

    # Exception: list of element
    sim = CausalBasedSimulator()
    changing_models = [
        "x0"
    ]
    try:
        sim._check_changing_models(changing_models, categorical_info, train_results)
    except RuntimeError:
        pass
    else:
        raise AssertionError

    # Exception: lack of key
    sim = CausalBasedSimulator()
    changing_models = [
        {
            "condition": None,
            "model": x0_model
        },
 
    ]
    try:
        sim._check_changing_models(changing_models, categorical_info, train_results)
    except RuntimeError:
        pass
    else:
        raise AssertionError

    # Exception: lask of key
    sim = CausalBasedSimulator()
    changing_models = [
        {
            "name": "x0",
            "model": x0_model
        },
 
    ]
    try:
        sim._check_changing_models(changing_models, categorical_info, train_results)
    except RuntimeError:
        pass
    else:
        raise AssertionError

    # Exception: lackk of key
    sim = CausalBasedSimulator()
    changing_models = [
        {
            "name": "x0",
            "condition": None,
        },
 
    ]
    try:
        sim._check_changing_models(changing_models, categorical_info, train_results)
    except RuntimeError:
        pass
    else:
        raise AssertionError

    # Exception: non existent variable name
    sim = CausalBasedSimulator()
    changing_models = [
        {
            "name": "x6",
            "condition": None,
            "model": x0_model
        },
    ]
    try:
        sim._check_changing_models(changing_models, categorical_info, train_results)
    except RuntimeError:
        pass
    else:
        raise AssertionError

    # Exception: on existent condition
    sim = CausalBasedSimulator()
    changing_models = [
        {
            "name": "x1",
            "condition": {"x2":"c"},
            "model": x1_model_a
        },
    ]
    try:
        sim._check_changing_models(changing_models, categorical_info, train_results)
    except RuntimeError:
        pass
    else:
        raise AssertionError

    # Exception: model is None
    sim = CausalBasedSimulator()
    changing_models = [
        {
            "name": "x1",
            "condition": None,
            "model": None
        },
    ]
    try:
        sim._check_changing_models(changing_models, categorical_info, train_results)
    except RuntimeError:
        pass
    else:
        raise AssertionError

    # Exception: wrong task type
    sim = CausalBasedSimulator()
    changing_models = [
        {
            "name": "x1",
            "condition": None,
            "model": x2_model
        },
    ]
    try:
        sim._check_changing_models(changing_models, categorical_info, train_results)
    except RuntimeError:
        pass
    else:
        raise AssertionError

    # Exception: wrong model type
    sim = CausalBasedSimulator()
    changing_models = [
        {
            "name": "x0",
            "condition": None,
            "model": x2_model
        },
    ]
    try:
        sim._check_changing_models(changing_models, categorical_info, train_results)
    except RuntimeError:
        pass
    else:
        raise AssertionError

def test_get_causal_order(init, test_data):
    init()
    X, causal_graph = test_data

    # Normal
    sim = CausalBasedSimulator()
    ret = sim._get_causal_order(causal_graph)
    assert ret.tolist() == [3, 0, 2, 1, 4, 5]

def _read_train_result_attrs(name, sim):
    fname = f"{DATA_DIR_PATH}/{name}.pickle"

    if ENABLE_CAPTURE:
        with open(fname, "wb") as f:
            pickle.dump((sim.residual_, sim.train_result_), f, protocol=3)

    with open(fname, "rb") as f:
        residual_, train_result_ = pickle.load(f)

    return residual_, train_result_

def _is_same_data(df, df2, exclude_cols=[], nan_cols=[]):
    if df.shape != df2.shape:
        raise AssertionError

    if (df.index != df2.index).any():
        raise AssertionError

    if (df.columns != df2.columns).any():
        raise AssertionError

    if (df.dtypes != df2.dtypes).any():
        raise AssertionError

    # numeric
    non_cat = df.dtypes != "category"
    non_cat2 = df2.dtypes != "category"
    if list(non_cat) != list(non_cat2):
        raise AssertionError

    removes = exclude_cols + nan_cols
    for c in removes:
        non_cat[list(df.columns).index(c)] = False
        non_cat2[list(df2.columns).index(c)] = False
    assert (np.isclose(df.loc[:, non_cat], df2.loc[:, non_cat2])).all()

    # categorical
    cat = df.dtypes == "category"
    cat2 = df2.dtypes == "category"
    if list(cat) != list(cat2):
        raise AssertionError
    if len(cat) == 0 and len(cat2) == 0:
        assert np.all(df.loc[:, cat] == df2.loc[:, cat2])

def _is_same_coef_intercept(train_result, train_result2):
    if len(set(train_result.keys()) - set(train_result2)) != 0:
        raise AssertionError

    for column in train_result.keys():
        for cond_result, cond_result2 in zip(train_result[column], train_result2[column]):
            model = cond_result["model"]
            model2 = cond_result2["model"]

            if model is None and model2 is None:
                continue

            assert np.isclose(model.coef_, model2.coef_).all()
            assert np.isclose(model.intercept_, model2.intercept_).all()

def test_train(init, test_data, test_data2, test_data3):
    init()

    X, causal_graph = test_data
    X2, causal_graph2 = test_data2
    X3, causal_graph3 = test_data3

    # Normal
    sim = CausalBasedSimulator()
    sim.train(X, causal_graph)

    residual_, train_result_ = _read_train_result_attrs("test_train", sim)

    _is_same_data(residual_, sim.residual_, nan_cols=["x3"])
    _is_same_coef_intercept(train_result_, sim.train_result_)

    # Normal2
    sim = CausalBasedSimulator()
    models = {
        "x0": SVR(kernel="linear"),
        "x2": SVR(kernel="linear"),
        "x4": SVR(kernel="linear"),
    }
    sim.train(X, causal_graph, models=models)

    residual_, train_result_ = _read_train_result_attrs("test_train2", sim)

    _is_same_data(residual_, sim.residual_, nan_cols=["x3"])
    _is_same_coef_intercept(train_result_, sim.train_result_)

    # Normal3
    sim = CausalBasedSimulator()
    sim.train(X2, causal_graph2)

    residual_, train_result_ = _read_train_result_attrs("test_train3", sim)

    _is_same_data(residual_, sim.residual_, nan_cols=["x2", "x3"])
    _is_same_coef_intercept(train_result_, sim.train_result_)

    # Normal4
    sim = CausalBasedSimulator()
    sim.train(X3, causal_graph3)

    residual_, train_result_ = _read_train_result_attrs("test_train4", sim)

    _is_same_data(residual_, sim.residual_, nan_cols=["x0", "x2"])

    assert train_result_["x1"][0]["model"].expected_value_ ==  sim.train_result_["x1"][0]["model"].expected_value_
    assert train_result_["x1"][1]["model"].expected_value_ ==  sim.train_result_["x1"][1]["model"].expected_value_
    assert list(train_result_["x2"][0]["model"].classes_) == list(sim.train_result_["x2"][0]["model"].classes_)
    assert list(train_result_["x2"][1]["model"].classes_) == list(sim.train_result_["x2"][1]["model"].classes_)
    assert list(train_result_["x2"][0]["model"].p_) == list(sim.train_result_["x2"][0]["model"].p_)
    assert list(train_result_["x2"][1]["model"].p_) == list(sim.train_result_["x2"][1]["model"].p_)

def test_run(init, test_data, test_data2, test_data3):
    init()

    X, causal_graph = test_data
    X2, causal_graph2 = test_data2
    X3, causal_graph3 = test_data3

    # Normal: no optios
    sim = CausalBasedSimulator()
    sim.train(X, causal_graph)
    sim.run()

    if ENABLE_CAPTURE:
        sim.simulated_data_.to_csv(f"{DATA_DIR_PATH}/test_run-simulated_data.csv", index=None)

    truth_df = pd.read_csv(f"{DATA_DIR_PATH}/test_run-simulated_data.csv")
    _is_same_data(sim.simulated_data_, truth_df)

    # Normal: only exogeous
    sim = CausalBasedSimulator()
    sim.train(X, causal_graph)
    changing_exog = {
        "x0": 2 * X["x0"].copy(),
    }
    sim.run(changing_exog=changing_exog)

    if ENABLE_CAPTURE:
        sim.simulated_data_.to_csv(f"{DATA_DIR_PATH}/test_run2-simulated_data.csv", index=None)

    truth_df = pd.read_csv(f"{DATA_DIR_PATH}/test_run2-simulated_data.csv")
    _is_same_data(sim.simulated_data_, truth_df)

    # Normal: only models
    sim = CausalBasedSimulator()
    sim.train(X, causal_graph)
    train_x = X[sim.train_result_["x2"][0]["exp_columns"]].copy()
    train_x["x3"] /= 4
    train_y = X["x2"]
    model = LinearRegression()
    model.fit(train_x.values, train_y.values)
    changing_models = [
        {
            "name": "x2",
            "condition": None,
            "model": model
        }
    ]
    sim.run(changing_models=changing_models)

    if ENABLE_CAPTURE:
        sim.simulated_data_.to_csv(f"{DATA_DIR_PATH}/test_run3-simulated_data.csv", index=None)

    truth_df = pd.read_csv(f"{DATA_DIR_PATH}/test_run3-simulated_data.csv")
    _is_same_data(sim.simulated_data_, truth_df)

    # Normal: exogenous and models
    sim = CausalBasedSimulator()
    sim.train(X, causal_graph)
    exp_columns_x2 = sim.train_result_["x2"][0]["exp_columns"]
    train_x = X[exp_columns_x2].copy()
    train_x["x3"] /= 4
    train_y = X["x2"]
    model = LinearRegression()
    model.fit(train_x.values, train_y.values)
    changing_models = [
        {
            "name": "x2",
            "condition": None,
            "model": model
        }
    ]
    changing_exog = {
        "x2": train_y - model.predict(X[exp_columns_x2].values)
    }
    sim.run(changing_exog=changing_exog, changing_models=changing_models)

    if ENABLE_CAPTURE:
        sim.simulated_data_.to_csv(f"{DATA_DIR_PATH}/test_run4-simulated_data.csv", index=None)

    truth_df = pd.read_csv(f"{DATA_DIR_PATH}/test_run4-simulated_data.csv")
    _is_same_data(sim.simulated_data_, truth_df)

    # Normal: no options
    sim = CausalBasedSimulator()
    sim.train(X2, causal_graph2)
    sim.run()

    if ENABLE_CAPTURE:
        sim.simulated_data_.to_csv(f"{DATA_DIR_PATH}/test_run5-simulated_data.csv", index=None)

    truth_df = pd.read_csv(f"{DATA_DIR_PATH}/test_run5-simulated_data.csv")
    truth_df["x2"] = truth_df["x2"].astype("category")
    _is_same_data(sim.simulated_data_, truth_df)

    # Normal: no optios
    sim = CausalBasedSimulator()
    sim.train(X3, causal_graph3)
    sim.run()

    if ENABLE_CAPTURE:
        sim.simulated_data_.to_csv(f"{DATA_DIR_PATH}/test_run6-simulated_data.csv", index=None)

    truth_df = pd.read_csv(f"{DATA_DIR_PATH}/test_run6-simulated_data.csv")
    truth_df["x0"] = truth_df["x0"].astype("category")
    truth_df["x2"] = truth_df["x2"].astype("category")
    _is_same_data(sim.simulated_data_, truth_df)
