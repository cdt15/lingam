import os

import numpy as np
import pandas as pd

from lingam.longitudinal_lingam import LongitudinalLiNGAM


def test_fit_success():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 0.7*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.3*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.3*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.7*x1 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.5*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.5*x1 + np.random.uniform(size=1000)
    X3 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = np.empty((3, 1000, 4))
    X_list[0] = X1
    X_list[1] = X2
    X_list[2] = X3

    model = LongitudinalLiNGAM()
    model.fit(X_list)

    # check causal ordering
    cos = model.causal_orders_
    for co in cos[1:]:
        assert co.index(0) < co.index(1) < co.index(3)

    # check B(t,t)
    B_t = model.adjacency_matrices_[1, 0] # B(1,1)
    assert B_t[1, 0] > 0.2 and B_t[3, 1] > 0.6
    B_t[1, 0] = 0.0
    B_t[3, 1] = 0.0
    assert np.sum(B_t) < 0.1

    B_t = model.adjacency_matrices_[2, 0] # B(2,2)
    assert B_t[1, 0] > 0.4 and B_t[3, 1] > 0.4
    B_t[1, 0] = 0.0
    B_t[3, 1] = 0.0
    assert np.sum(B_t) < 0.1

    # check B(t,t-τ)
    B_tau = model.adjacency_matrices_[1, 1] # B(1,0)
    assert B_tau[0, 2] > 0.4 and B_tau[2, 3] > 0.4

    B_tau = model.adjacency_matrices_[1, 1] # B(2,1)
    assert B_tau[0, 2] > 0.4 and B_tau[2, 3] > 0.4

    # fit by list
    X_list = [X1, X2, X3]
    model = LongitudinalLiNGAM()
    model.fit(X_list)


def test_fit_invalid_data():
    # Different features
    x0 = np.random.uniform(size=1000)
    x1 = 0.7*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.3*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.3*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    X2 = pd.DataFrame(np.array([x0, x1, x2]).T,
                      columns=['x0', 'x1', 'x2'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.5*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.5*x1 + np.random.uniform(size=1000)
    X3 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2, X3]

    try:
        model = LongitudinalLiNGAM()
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Not list data
    X = 1
    try:
        model = LongitudinalLiNGAM()
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include not-array data
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, 1]

    try:
        model = LongitudinalLiNGAM()
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include non-numeric data
    x0 = np.random.uniform(size=1000)
    x1 = 0.7*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.3*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = np.array(['X']*1000) # <== non-numeric
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = np.array(['X']*1000) # <== non-numeric
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.5*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.5*x1 + np.random.uniform(size=1000)
    X3 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2, X3]

    try:
        model = LongitudinalLiNGAM()
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include NaN values
    x0 = np.random.uniform(size=1000)
    x1 = 0.7*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.3*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.3*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.7*x1 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])
    X2.iloc[100, 0] = np.nan # set nan

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.5*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.5*x1 + np.random.uniform(size=1000)
    X3 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2, X3]

    try:
        model = LongitudinalLiNGAM()
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include infinite values
    x0 = np.random.uniform(size=1000)
    x1 = 0.7*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.3*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.3*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.7*x1 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])
    X2.iloc[100, 0] = np.inf # set inf

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.5*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.5*x1 + np.random.uniform(size=1000)
    X3 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2, X3]

    try:
        model = LongitudinalLiNGAM()
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

def test_bootstrap_success():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 0.7*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.3*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.3*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.7*x1 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.5*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.5*x1 + np.random.uniform(size=1000)
    X3 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = np.empty((3, 1000, 4))
    X_list[0] = X1
    X_list[1] = X2
    X_list[2] = X3

    model = LongitudinalLiNGAM()
    model.bootstrap(X_list, n_sampling=3)

    # fit by list
    X_list = [X1, X2, X3]
    model = LongitudinalLiNGAM()
    model.bootstrap(X_list, n_sampling=3)

def test_bootstrap_invalid_data():
    # Different features
    x0 = np.random.uniform(size=1000)
    x1 = 0.7*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.3*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.3*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    X2 = pd.DataFrame(np.array([x0, x1, x2]).T,
                      columns=['x0', 'x1', 'x2'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.5*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.5*x1 + np.random.uniform(size=1000)
    X3 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2, X3]

    try:
        model = LongitudinalLiNGAM()
        model.bootstrap(X_list, n_sampling=3)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Not list data
    X = 1
    try:
        model = LongitudinalLiNGAM()
        model.bootstrap(X, n_sampling=3)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include not-array data
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, 1]

    try:
        model = LongitudinalLiNGAM()
        model.bootstrap(X, n_sampling=3)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include non-numeric data
    x0 = np.random.uniform(size=1000)
    x1 = 0.7*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.3*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = np.array(['X']*1000) # <== non-numeric
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = np.array(['X']*1000) # <== non-numeric
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.5*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.5*x1 + np.random.uniform(size=1000)
    X3 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2, X3]

    try:
        model = LongitudinalLiNGAM()
        model.bootstrap(X_list, n_sampling=3)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include NaN values
    x0 = np.random.uniform(size=1000)
    x1 = 0.7*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.3*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.3*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.7*x1 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])
    X2.iloc[100, 0] = np.nan # set nan

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.5*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.5*x1 + np.random.uniform(size=1000)
    X3 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2, X3]

    try:
        model = LongitudinalLiNGAM()
        model.bootstrap(X_list, n_sampling=3)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include infinite values
    x0 = np.random.uniform(size=1000)
    x1 = 0.7*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 0.3*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.3*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.7*x1 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])
    X2.iloc[100, 0] = np.inf # set inf

    x0 = np.random.uniform(size=1000) + 0.5*x2
    x1 = 0.5*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000) + 0.5*x3
    x3 = 0.5*x1 + np.random.uniform(size=1000)
    X3 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2, X3]

    try:
        model = LongitudinalLiNGAM()
        model.bootstrap(X_list, n_sampling=3)
    except ValueError:
        pass
    else:
        raise AssertionError
