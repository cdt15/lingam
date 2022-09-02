import numpy as np
import pandas as pd
from lingam.ica_lingam import ICALiNGAM


def test_fit_success():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0 * x1 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3]).T, columns=['x0', 'x1', 'x2', 'x3'])

    model = ICALiNGAM()
    model.fit(X)

    # check the causal ordering
    co = model.causal_order_
    assert co.index(0) < co.index(1) < co.index(3)

    # check the adjacency matrix
    am = model.adjacency_matrix_
    # assert am[1, 0] > 1.5 and am[3, 1] > 3.5

    am[1, 0] = 0.0
    am[3, 1] = 0.0
    # assert np.sum(am) < 0.1

    # for coverage
    matrix = np.array([
        [0, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ])
    model = ICALiNGAM()
    model._search_causal_order(matrix)

    # for coverage
    matrix = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
    ])
    model = ICALiNGAM()
    model._search_causal_order(matrix)


def test_fit_invalid_data():
    # Not array data
    X = 1
    try:
        model = ICALiNGAM()
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
        model = ICALiNGAM()
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include NaN values
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])
    X.iloc[100, 0] = np.nan
    try:
        model = ICALiNGAM()
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include infinite values
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])
    X.iloc[100, 0] = np.inf
    try:
        model = ICALiNGAM()
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError
