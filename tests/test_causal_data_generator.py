import pytest

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from scipy.special import expit

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingam.experimental import CausalDataGenerator

@pytest.fixture
def init():
    return lambda :np.random.seed(0)

@pytest.fixture
def test_data():
    np.random.seed(0)

    N=10000

    errors = pd.DataFrame({
        "x1": np.random.uniform(size=N),
        "x2": np.random.uniform(size=N),
        "x3": np.random.uniform(size=N),
        "x4": np.random.uniform(size=N),
    })
    
    def model_x1(df, error, args):
        return 2 * df["x3"] - 2 * df["x4"] + error
    
    def model_x2(df, error, args):
        return 2 * df["x1"] - 3 * df["x3"] - 3 * df["x4"] + error
        
    def model_x3(df, error, args):
        return error
    
    def model_x4(df, error, args):
        return error
    
    models = {
        "x1": model_x1,
        "x2": model_x2,
        "x3": model_x3,
        "x4": model_x4,
    }
    
    causal_order = ["x3", "x4", "x1", "x2"]
    
    X = pd.DataFrame(np.nan, index=np.arange(N), columns=list(errors.keys()))
    for name in causal_order:
        X[name] = models[name](X, errors[name], None)

    return X, models, errors, causal_order

@pytest.fixture
def test_data2():
    np.random.seed(0)

    N=10000

    errors = pd.DataFrame({
        "x1": np.random.uniform(size=N),
        "x2": np.random.uniform(size=N),
        "x3": np.random.choice(["a", "b"], size=N),
        "x4": np.random.uniform(size=N),
    })
    
    def model_x1(df, error, args):
        def func(s):
            return 1.5 * s["x4"] + (-2 if s["x3"] == "a" else 0.5) + errors.loc[s.name, "x1"]
        
        p = df[["x3", "x4"]].apply(func, axis=1)
        d = expit(p).apply(lambda v: "c" if v > 0.5 else "d")
        return pd.Categorical(d)
            
    def model_x2(df, error, args):
        w_map = {
            ("a", "c"): 2,
            ("a", "d"): -2,
            ("b", "c"): 0.5,
            ("b", "d"): -0.5,
        }
        def func(s):
            return w_map[s["x3"], s["x1"]] * s["x4"]
        
        d = df[["x1", "x3", "x4"]].apply(func, axis=1) + error
        return d
        
    def model_x3(df, error, args):
        return pd.Categorical(errors["x3"])
    
    def model_x4(df, error, args):
        return errors["x4"]
    
    models = {
        "x1": model_x1,
        "x2": model_x2,
        "x3": model_x3,
        "x4": model_x4,
    }
    
    causal_order = ["x3", "x4", "x1", "x2"]
    
    X = pd.DataFrame(np.nan, index=np.arange(N), columns=list(errors.keys()))
    for name in causal_order:
        X[name] = models[name](X, errors[name], None)
    
    return X, models, errors, causal_order

def test_continuous(init, test_data):
    init()

    X, models, errors, causal_order = test_data

    g = CausalDataGenerator()
    g.fit(X, "x1", "x2", adjustments=["x3", "x4"], model_cause=LinearRegression(), model_effect=LinearRegression())

    # generate data using ground truth functions and noises.
    generated = g.generate(interv_exog=errors.to_dict(orient="list"), cause_model=models["x1"], effect_model=models["x2"])
    mae = mean_absolute_error(X, generated)
    if not np.isclose(mae, 0):
        raise ValueError

def test_discrete(init, test_data2):
    init()

    X, models, errors, causal_order = test_data2

    # models
    transformers = [
        ("categorical", OneHotEncoder(sparse_output=False), ["x3"]),
        ("numeric", "passthrough", ["x4"]),
    ]
    trans = ColumnTransformer(transformers=transformers)
    
    model_cause = Pipeline([
        ("transformer", trans),
        ("estimator", LogisticRegression())
    ])
    
    transformers = [
        ("categorical", OneHotEncoder(sparse_output=False), ["x1", "x3"]),
        ("numeric", "passthrough", ["x4"]),
    ]
    trans = ColumnTransformer(transformers=transformers)
    
    model_effect = Pipeline([
        ("transformer", trans),
        ("estimator", LinearRegression())
    ])

    g = CausalDataGenerator()
    g.fit(X, "x1", "x2", adjustments=["x3", "x4"], model_cause=model_cause, model_effect=model_effect)

    # generate data using ground truth functions and noises.
    generated = g.generate(interv_exog=errors.to_dict(orient="list"), cause_model=models["x1"], effect_model=models["x2"])
    mae = mean_absolute_error(X[["x2", "x4"]], generated[["x2", "x4"]])
    if not np.isclose(mae, 0):
        raise ValueError

    s = (X[["x1", "x3"]] == generated[["x1", "x3"]]).sum().sum()
    if s != X.shape[0] * 2:
        raise ValueError
