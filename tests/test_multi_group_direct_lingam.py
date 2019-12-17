import os

import numpy as np
import pandas as pd

from lingam.multi_group_direct_lingam import MultiGroupDirectLiNGAM


def test_fit_success():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000)
    x1 = 2.1*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.2*x1 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2]

    model = MultiGroupDirectLiNGAM()
    model.fit(X_list)

    # check the causal ordering
    co = model.causal_order_
    assert co.index(0) < co.index(1) < co.index(3)

    # check the adjacency matrix
    am = model.adjacency_matrices_[0]
    assert am[1, 0] > 1.5 and am[3, 1] > 3.5

    # check the adjacency matrix
    am = model.adjacency_matrices_[1]
    assert am[1, 0] > 1.5 and am[3, 1] > 3.5

    am[1, 0] = 0.0
    am[3, 1] = 0.0
    assert np.sum(am) < 0.1


def test_fit_invalid_data():
    # Different features
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000)
    x1 = 2.1*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2]).T,
                      columns=['x0', 'x1', 'x2'])

    X_list = [X1, X2]

    try:
        model = MultiGroupDirectLiNGAM()
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Not list data
    X = 1
    try:
        model = MultiGroupDirectLiNGAM()
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
        model = MultiGroupDirectLiNGAM()
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include non-numeric data
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=5)
    x1 = np.array(['X', 'Y', 'X', 'Y', 'X'])
    X2 = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])

    X_list = [X1, X2]

    try:
        model = MultiGroupDirectLiNGAM()
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include NaN values
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])
    X2.iloc[100, 0] = np.nan

    X_list = [X1, X2]

    try:
        model = MultiGroupDirectLiNGAM()
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include infinite values
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])
    X.iloc[100, 0] = np.inf

    X_list = [X1, X2]

    try:
        model = MultiGroupDirectLiNGAM()
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError


def test_prior_knowledge_success():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000)
    x1 = 2.1*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.2*x1 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2]

    # prior knowledge: nothing
    pk = np.array([
        [0, -1, -1, -1],
        [-1,  0, -1, -1],
        [-1, -1,  0, -1],
        [-1, -1, -1,  0],
    ])

    model = MultiGroupDirectLiNGAM(prior_knowledge=pk)
    model.fit(X_list)
    co = model.causal_order_
    assert co.index(0) < co.index(1) < co.index(3)

    # prior knowledge: x1 is exogenous
    pk = np.array([
        [0, -1, -1, -1],
        [0,  0,  0,  0],
        [-1, -1,  0, -1],
        [-1, -1, -1,  0],
    ])

    model = MultiGroupDirectLiNGAM(prior_knowledge=pk)
    model.fit(X_list)
    co = model.causal_order_
    print(co)
    assert co.index(1) < co.index(0)
    assert co.index(1) < co.index(2)
    assert co.index(1) < co.index(3)

    # prior knowledge: x0 is sink
    pk = np.array([
        [0, -1, -1, -1],
        [0,  0, -1, -1],
        [0, -1,  0, -1],
        [0, -1, -1,  0],
    ])

    model = MultiGroupDirectLiNGAM(prior_knowledge=pk)
    model.fit(X_list)
    co = model.causal_order_
    print(co)
    assert co.index(0) > co.index(1)
    assert co.index(0) > co.index(2)
    assert co.index(0) > co.index(3)

    # prior knowledge: x2-->x3 has path
    pk = np.array([
        [0, -1, -1, -1],
        [-1,  0, -1, -1],
        [-1, -1,  0,  0],
        [-1, -1,  1,  0],
    ])

    model = MultiGroupDirectLiNGAM(prior_knowledge=pk)
    model.fit(X_list)
    co = model.causal_order_
    print(co)
    assert co.index(2) < co.index(3)

    # prior knowledge: x1-->x3 does not have path
    pk = np.array([
        [0, -1, -1, -1],
        [-1,  0, -1, -1],
        [-1, -1,  0, -1],
        [-1,  0, -1,  0],
    ])

    model = MultiGroupDirectLiNGAM(prior_knowledge=pk)
    model.fit(X_list)


def test_prior_knowledge_invalid():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0*x1 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    x0 = np.random.uniform(size=1000)
    x1 = 2.1*x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.2*x1 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                      columns=['x0', 'x1', 'x2', 'x3'])

    X_list = [X1, X2]

    # prior knowledge: invalid
    pk = np.array([
        [0, -1, -1],
        [-1,  0, -1],
        [-1, -1,  0],
        [-1, -1, -1],
    ])

    try:
        model = MultiGroupDirectLiNGAM(prior_knowledge=pk)
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError
