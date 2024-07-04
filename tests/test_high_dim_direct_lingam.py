import os
import warnings

import numpy as np
import pandas as pd
from lingam import HighDimDirectLiNGAM


def test_fit_success():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0 * x1 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3]).T, columns=['x0', 'x1', 'x2', 'x3'])

    model = HighDimDirectLiNGAM()
    model.fit(X)

    # causal order
    co = model.causal_order_
    assert co.index(0) < co.index(1) < co.index(3)

    # J
    model = HighDimDirectLiNGAM(J=6)
    model.fit(X)
    co = model.causal_order_
    assert co.index(0) < co.index(1) < co.index(3)

    # K
    model = HighDimDirectLiNGAM(K=2)
    model.fit(X)
    co = model.causal_order_
    assert co.index(0) < co.index(1) < co.index(3)

    # alpha
    model = HighDimDirectLiNGAM(alpha=0.1)
    model.fit(X)
    co = model.causal_order_
    assert co.index(0) < co.index(1) < co.index(3)

    # estimate_adj_mat
    model = HighDimDirectLiNGAM(estimate_adj_mat=False)
    model.fit(X)
    co = model.causal_order_
    assert co.index(0) < co.index(1) < co.index(3)
    assert model.adjacency_matrix_ is None

    # random_state
    model = HighDimDirectLiNGAM(random_state=0)
    model.fit(X)
    model2 = HighDimDirectLiNGAM(random_state=0)
    model2.fit(X)
    assert np.isclose(model.adjacency_matrix_, model2.adjacency_matrix_).all()

    # n_samples <= n_features
    X2 = np.random.uniform(-1, 1, size=(10, 10))

    model = HighDimDirectLiNGAM()
    with warnings.catch_warnings(record=True) as w:
        model.fit(X2)
        assert w[0].category == UserWarning

def test_fit_invalid_data():
    # Not array data
    X = 1
    try:
        model = HighDimDirectLiNGAM()
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
        model = HighDimDirectLiNGAM()
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
        model = HighDimDirectLiNGAM()
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
        model = HighDimDirectLiNGAM()
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # J > 2
    try:
        model = HighDimDirectLiNGAM(J=2)
    except ValueError:
        pass
    else:
        raise AssertionError

    # J must be integer
    try:
        model = HighDimDirectLiNGAM(J=4.0)
    except TypeError:
        pass
    else:
        raise AssertionError

    # K >= 1
    try:
        model = HighDimDirectLiNGAM(K=0)
    except ValueError:
        pass
    else:
        raise AssertionError

    # K must be integer
    try:
        model = HighDimDirectLiNGAM(K=2.0)
    except TypeError:
        pass
    else:
        raise AssertionError

    # 0 <= alpha <= 1
    try:
        model = HighDimDirectLiNGAM(alpha=-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError
