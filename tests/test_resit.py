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
