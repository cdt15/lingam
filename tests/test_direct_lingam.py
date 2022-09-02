import os

import numpy as np
import pandas as pd
from lingam.direct_lingam import DirectLiNGAM


def test_fit_success():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0 * x1 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3]).T, columns=['x0', 'x1', 'x2', 'x3'])

    model = DirectLiNGAM()
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

    # check estimate_total_effect (_BaseLiNGAM)
    te = model.estimate_total_effect(X, 0, 3)
    # assert te > 7.5 and te < 8.5

    te = model.estimate_total_effect(X, 1, 3)
    # assert te > 3.5 and te < 4.5

    te = model.estimate_total_effect(X, 0, 2)
    # assert te < 0.2

    te = model.estimate_total_effect(X, 3, 0)
    # assert te < 0.2

    p_values = model.get_error_independence_p_values(X)


def test_fit_invalid_data():
    # Not array data
    X = 1
    try:
        model = DirectLiNGAM()
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
        model = DirectLiNGAM()
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
        model = DirectLiNGAM()
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
        model = DirectLiNGAM()
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError


def test_prior_knowledge_success():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0 * x1 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3]).T, columns=['x0', 'x1', 'x2', 'x3'])

    # prior knowledge: nothing
    pk = np.array(
        [
            [0, -1, -1, -1],
            [-1, 0, -1, -1],
            [-1, -1, 0, -1],
            [-1, -1, -1, 0],
        ]
    )

    model = DirectLiNGAM(prior_knowledge=pk)
    model.fit(X)
    co = model.causal_order_
    assert co.index(0) < co.index(1) < co.index(3)

    # prior knowledge: x1 is exogenous
    pk = np.array(
        [
            [0, -1, -1, -1],
            [0, 0, 0, 0],
            [-1, -1, 0, -1],
            [-1, -1, -1, 0],
        ]
    )

    model = DirectLiNGAM(prior_knowledge=pk)
    model.fit(X)
    co = model.causal_order_
    print(co)
    assert co.index(1) < co.index(0)
    assert co.index(1) < co.index(2)
    assert co.index(1) < co.index(3)

    # prior knowledge: x0 is sink
    pk = np.array(
        [
            [0, -1, -1, -1],
            [0, 0, -1, -1],
            [0, -1, 0, -1],
            [0, -1, -1, 0],
        ]
    )

    model = DirectLiNGAM(prior_knowledge=pk)
    model.fit(X)
    co = model.causal_order_
    print(co)
    assert co.index(0) > co.index(1)
    assert co.index(0) > co.index(2)
    assert co.index(0) > co.index(3)

    # prior knowledge: x2-->x3 has path
    pk = np.array(
        [
            [0, -1, -1, -1],
            [-1, 0, -1, -1],
            [-1, -1, 0, 0],
            [-1, -1, 1, 0],
        ]
    )

    model = DirectLiNGAM(prior_knowledge=pk)
    model.fit(X)
    co = model.causal_order_
    print(co)
    assert co.index(2) < co.index(3)

    # prior knowledge: x1-->x3 does not have path
    pk = np.array(
        [
            [0, -1, -1, -1],
            [-1, 0, -1, -1],
            [-1, -1, 0, -1],
            [-1, 0, -1, 0],
        ]
    )

    model = DirectLiNGAM(prior_knowledge=pk)
    model.fit(X)

    # prior knowledge: x1 is exogenous (apply_prior_knowledge_softly)
    pk = np.array(
        [
            [0, -1, -1, -1],
            [0, 0, 0, 0],
            [-1, -1, 0, -1],
            [-1, -1, -1, 0],
        ]
    )

    model = DirectLiNGAM(prior_knowledge=pk, apply_prior_knowledge_softly=True)
    model.fit(X)
    co = model.causal_order_
    print(co)
    assert co.index(1) < co.index(0)
    assert co.index(1) < co.index(2)
    assert co.index(1) < co.index(3)

    # prior knowledge: x0 is sink (apply_prior_knowledge_softly)
    pk = np.array(
        [
            [0, -1, -1, -1],
            [0, 0, -1, -1],
            [0, -1, 0, -1],
            [0, -1, -1, 0],
        ]
    )

    model = DirectLiNGAM(prior_knowledge=pk, apply_prior_knowledge_softly=True)
    model.fit(X)
    co = model.causal_order_
    print(co)
    assert co.index(0) > co.index(1)
    assert co.index(0) > co.index(2)
    assert co.index(0) > co.index(3)

    # prior knowledge: x2-->x3 has path (apply_prior_knowledge_softly)
    pk = np.array(
        [
            [0, -1, -1, -1],
            [-1, 0, -1, -1],
            [-1, -1, 0, 0],
            [-1, -1, 1, 0],
        ]
    )

    model = DirectLiNGAM(prior_knowledge=pk, apply_prior_knowledge_softly=True)
    model.fit(X)
    co = model.causal_order_
    print(co)
    assert co.index(2) < co.index(3)

    # prior knowledge: x1-->x3 does not have path (apply_prior_knowledge_softly)
    pk = np.array(
        [
            [0, -1, -1, -1],
            [-1, 0, -1, -1],
            [-1, -1, 0, -1],
            [-1, 0, -1, 0],
        ]
    )

    model = DirectLiNGAM(prior_knowledge=pk, apply_prior_knowledge_softly=True)
    model.fit(X)

    # prior knowledge: x1 is exogenous (apply_prior_knowledge_softly)
    pk = np.array(
        [
            [0, -1, -1, 0],
            [0, 0, -1, 0],
            [0, -1, 0, -1],
            [0, 0, 0, 0],
        ]
    )

    model = DirectLiNGAM(prior_knowledge=pk, apply_prior_knowledge_softly=True)
    model.fit(X)
    co = model.causal_order_


def test_prior_knowledge_invalid():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0 * x1 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3]).T, columns=['x0', 'x1', 'x2', 'x3'])

    # prior knowledge: invalid
    pk = np.array(
        [
            [0, -1, 1],
            [-1, 0, -1],
            [-1, -1, 0],
            [-1, -1, -1],
        ]
    )

    try:
        model = DirectLiNGAM(prior_knowledge=pk)
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # prior knowledge: inconsistent
    pk = np.array(
        [
            [0, 1, -1, -1],
            [1, 0, -1, -1],
            [-1, -1, 0, -1],
            [-1, -1, -1, 0],
        ]
    )

    try:
        model = DirectLiNGAM(prior_knowledge=pk)
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError


def test_kernel_mode():
    # causal direction: x0 --> x1 --> x2
    x0 = np.random.uniform(size=1100)
    x1 = 0.5 * x0 + np.random.uniform(size=1100)
    x2 = 0.5 * x0 + 0.5 * x1 + np.random.uniform(size=1100)
    X = pd.DataFrame(np.array([x0, x1, x2]).T, columns=['x0', 'x1', 'x2'])

    model = DirectLiNGAM(measure='kernel')
    model.fit(X.iloc[:100, :])

    model = DirectLiNGAM(measure='kernel')
    model.fit(X)

def test_base():
    # causal direction: x0 --> x1 --> x2
    x0 = np.random.uniform(size=1100)
    x1 = 0.5 * x0 + np.random.uniform(size=1100)
    x2 = 0.5 * x0 + 0.5 * x1 + np.random.uniform(size=1100)
    X = pd.DataFrame(np.array([x0, x1, x2]).T, columns=['x0', 'x1', 'x2'])

    # prior knowledge: x1 is exogenous
    pk = np.array(
        [
            [0, 0, 0, 0],
            [-1, 0, -1, -1],
            [-1, -1, 0, -1],
            [-1, -1, -1, 0],
        ]
    )
    model = DirectLiNGAM()
    model._causal_order = [1, 0, 2]
    model._estimate_adjacency_matrix(X.values, prior_knowledge=pk)
