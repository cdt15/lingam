import os

import numpy as np
import pandas as pd

from lingam.bottom_up_parce_lingam import BottomUpParceLiNGAM

def test_fit_success():
    # causal direction: x5 --> x1, x3 --> x0 --> x2 --> x4
    n_samples = 1000
    x5 = np.random.uniform(size=n_samples)
    x1 = 2.0*x5 + np.random.uniform(size=n_samples)
    x3 = 2.0*x5 + np.random.uniform(size=n_samples)
    x0 = 0.5*x1 - 0.5*x3 + np.random.uniform(size=n_samples)
    x2 = 0.5*x0 + np.random.uniform(size=n_samples)
    x4 = 1.0*x2 + np.random.uniform(size=n_samples)
    # x5 is latent confounders
    X = pd.DataFrame(np.array([x0, x1, x2, x3, x4]).T,
                     columns=['x0', 'x1', 'x2', 'x3', 'x4'])

    model = BottomUpParceLiNGAM()
    model.fit(X)

    # check the causal ordering
    co = model.causal_order_
    flatten = lambda arr: [val for item in arr for val in (flatten(item) if hasattr(item, '__iter__') and not isinstance(item, str) else [item])]
    co = flatten(co)
    assert co.index(1) < co.index(0) < co.index(2) < co.index(4)
    assert co.index(3) < co.index(0) < co.index(2) < co.index(4)

    # check the adjacency matrix
    am = model.adjacency_matrix_
    assert am[2, 0] > 0
    assert am[4, 2] > 0

    am[0, 1] = 0.0
    am[0, 3] = 0.0
    am[1, 3] = 0.0
    am[3, 1] = 0.0
    am[2, 0] = 0.0
    am[4, 2] = 0.0
    assert np.sum(np.nan_to_num(am)) < 0.1

def test_fit_invalid_data():
    # Not array data
    X = 1
    try:
        model = BottomUpParceLiNGAM()
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
        model = BottomUpParceLiNGAM()
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include NaN values
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])
    X.iloc[100, 0] = np.nan
    try:
        model = BottomUpParceLiNGAM()
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include infinite values
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])
    X.iloc[100, 0] = np.inf
    try:
        model = BottomUpParceLiNGAM()
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

if __name__ == "__main__":
    test_fit_success()
    test_fit_invalid_data()
