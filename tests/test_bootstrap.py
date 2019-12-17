import os

import numpy as np
import pandas as pd

from lingam.bootstrap import BootstrapMixin, BootstrapResult


class DummyBoostrapMixin(BootstrapMixin):
    def __init__(self, random_state=None):
        self._random_state = random_state
        self._causal_order = None
        self._adjacency_matrix = None
        self._adjacency_matrices = [
            # causal direction: x0 --> x1 --> x2 --> x3
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            # causal direction: x1 --> x2 --> x3
            # However, a small causal effect is included
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.1, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            # causal direction: x2 --> x3
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            # causal direction: None
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            # causal direction: x0 --> x1 --> x2 --> x3
            # However, a negative causal effect is included
            [
                [0.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, -0.5, 0.0],
            ],
        ]

    def fit(self, X):
        self._adjacency_matrix = np.array(self._adjacency_matrices.pop(0))
        return self

    @property
    def causal_order_(self):
        return self._causal_order

    @property
    def adjacency_matrix_(self):
        return self._adjacency_matrix


class DummyBoostrapMixinNonCausal(BootstrapMixin):
    def __init__(self, random_state=None):
        self._random_state = random_state
        self._causal_order = None
        self._adjacency_matrix = None
        self._adjacency_matrices = [
            # causal direction: None
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            # causal direction: None
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            # causal direction: None
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            # causal direction: None
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            # causal direction: None
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        ]

    def fit(self, X):
        self._adjacency_matrix = np.array(self._adjacency_matrices.pop(0))
        return self

    @property
    def causal_order_(self):
        return self._causal_order

    @property
    def adjacency_matrix_(self):
        return self._adjacency_matrix


def test_bootstrap_success():
    x0 = np.random.uniform(size=1000)
    x1 = np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3]).T,
                     columns=['x0', 'x1', 'x2', 'x3'])

    model = DummyBoostrapMixin()
    result = model.bootstrap(X, n_sampling=5)

    # No argument
    cdc = result.get_causal_direction_counts()
    assert cdc['from'] == [2, 1, 0]
    assert cdc['to'] == [3, 2, 1]
    assert cdc['count'] == [4, 3, 2]

    # n_directions=2
    cdc = result.get_causal_direction_counts(n_directions=2)
    assert cdc['from'] == [2, 1]
    assert cdc['to'] == [3, 2]
    assert cdc['count'] == [4, 3]

    # min_causal_effect=0.2
    cdc = result.get_causal_direction_counts(min_causal_effect=0.2)
    assert cdc['from'] == [2, 0, 1]
    assert cdc['to'] == [3, 1, 2]
    assert cdc['count'] == [4, 2, 2]

    # min_causal_effect=0.6
    cdc = result.get_causal_direction_counts(min_causal_effect=0.6)
    assert cdc['from'] == [2, 0, 1]
    assert cdc['to'] == [3, 1, 2]
    assert cdc['count'] == [3, 2, 2]

    # split_by_causal_effect_sign=True
    cdc = result.get_causal_direction_counts(split_by_causal_effect_sign=True)
    assert cdc['from'] == [1, 2, 0, 2]
    assert cdc['to'] == [2, 3, 1, 3]
    assert cdc['count'] == [3, 3, 2, 1]
    assert cdc['sign'] == [1, 1, 1, -1]

    # No argument
    dagc = result.get_directed_acyclic_graph_counts()
    assert len(dagc['dag']) == 4
    assert dagc['dag'][0]['from'] == [0, 1, 2]
    assert dagc['dag'][0]['to'] == [1, 2, 3]
    assert dagc['count'][0] == 2
    assert not dagc['dag'][1]['from']
    assert not dagc['dag'][1]['to']
    assert dagc['count'][1] == 1
    assert dagc['dag'][2]['from'] == [2]
    assert dagc['dag'][2]['to'] == [3]
    assert dagc['count'][2] == 1
    assert dagc['dag'][3]['from'] == [1, 2]
    assert dagc['dag'][3]['to'] == [2, 3]
    assert dagc['count'][3] == 1

    # n_dags=2
    dagc = result.get_directed_acyclic_graph_counts(n_dags=2)
    assert len(dagc['dag']) == 2
    assert dagc['dag'][0]['from'] == [0, 1, 2]
    assert dagc['dag'][0]['to'] == [1, 2, 3]
    assert dagc['count'][0] == 2
    assert not dagc['dag'][1]['from']
    assert not dagc['dag'][1]['to']
    assert dagc['count'][1] == 1

    # min_causal_effect=0.2
    dagc = result.get_directed_acyclic_graph_counts(min_causal_effect=0.2)
    assert len(dagc['dag']) == 3
    assert dagc['dag'][0]['from'] == [2]
    assert dagc['dag'][0]['to'] == [3]
    assert dagc['count'][0] == 2
    assert dagc['dag'][1]['from'] == [0, 1, 2]
    assert dagc['dag'][1]['to'] == [1, 2, 3]
    assert dagc['count'][1] == 2
    assert not dagc['dag'][2]['from']
    assert not dagc['dag'][2]['to']
    assert dagc['count'][2] == 1

    # min_causal_effect=0.6
    dagc = result.get_directed_acyclic_graph_counts(min_causal_effect=0.6)
    assert len(dagc['dag']) == 4
    assert dagc['dag'][0]['from'] == [2]
    assert dagc['dag'][0]['to'] == [3]
    assert dagc['count'][0] == 2
    assert not dagc['dag'][1]['from']
    assert not dagc['dag'][1]['to']
    assert dagc['count'][1] == 1
    assert dagc['dag'][2]['from'] == [0, 1]
    assert dagc['dag'][2]['to'] == [1, 2]
    assert dagc['count'][2] == 1
    assert dagc['dag'][3]['from'] == [0, 1, 2]
    assert dagc['dag'][3]['to'] == [1, 2, 3]
    assert dagc['count'][3] == 1

    # split_by_causal_effect_sign=True
    dagc = result.get_directed_acyclic_graph_counts(split_by_causal_effect_sign=True)
    assert len(dagc['dag']) == 5
    print(dagc)
    assert not dagc['dag'][0]['from']
    assert not dagc['dag'][0]['to']
    assert not dagc['dag'][0]['sign']
    assert dagc['count'][0] == 1

    assert dagc['dag'][1]['from'] == [2]
    assert dagc['dag'][1]['to'] == [3]
    assert dagc['dag'][1]['sign'] == [1]
    assert dagc['count'][1] == 1

    assert dagc['dag'][2]['from'] == [1, 2]
    assert dagc['dag'][2]['to'] == [2, 3]
    assert dagc['dag'][2]['sign'] == [1, 1]
    assert dagc['count'][2] == 1

    assert dagc['dag'][3]['from'] == [0, 1, 2]
    assert dagc['dag'][3]['to'] == [1, 2, 3]
    assert dagc['dag'][3]['sign'] == [1, 1, -1]
    assert dagc['count'][3] == 1

    assert dagc['dag'][4]['from'] == [0, 1, 2]
    assert dagc['dag'][4]['to'] == [1, 2, 3]
    assert dagc['dag'][4]['sign'] == [1, 1, 1]
    assert dagc['count'][4] == 1



def test_bootstrap_invalid_data():
    # Not array data
    X = 1
    try:
        model = DummyBoostrapMixin()
        result = model.bootstrap(X, n_sampling=5)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include non-numeric data
    x0 = np.random.uniform(size=5)
    x1 = np.array(['X', 'Y', 'X', 'Y', 'X'])
    X = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])
    try:
        model = DummyBoostrapMixin()
        result = model.bootstrap(X, n_sampling=5)
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
        model = DummyBoostrapMixin()
        result = model.bootstrap(X, n_sampling=5)
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
        model = DummyBoostrapMixin()
        result = model.bootstrap(X, n_sampling=5)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Non-causal data
    x0 = np.random.uniform(size=1)
    x1 = np.random.uniform(size=1)
    X = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])
    model = DummyBoostrapMixinNonCausal()
    result = model.bootstrap(X, n_sampling=5)

    cdc = result.get_causal_direction_counts()
    assert not cdc['from'] and not cdc['to'] and not cdc['count']

    dagc = result.get_directed_acyclic_graph_counts()
    assert not dagc['dag'][0]['from'] and not dagc['dag'][0]['to'] and dagc['count'][0] == 5

    # Invalid argument: get_causal_direction_counts(n_directions=-1)
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])
    model = DummyBoostrapMixin()
    result = model.bootstrap(X, n_sampling=5)
    try:
        result.get_causal_direction_counts(n_directions=-1)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: get_causal_direction_counts(min_causal_effect=-1.0)
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])
    model = DummyBoostrapMixin()
    result = model.bootstrap(X, n_sampling=5)
    try:
        result.get_causal_direction_counts(min_causal_effect=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: get_directed_acyclic_graph_counts(n_dags=-1)
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])
    model = DummyBoostrapMixin()
    result = model.bootstrap(X, n_sampling=5)
    try:
        result.get_directed_acyclic_graph_counts(n_dags=-1)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid argument: get_directed_acyclic_graph_counts(min_causal_effect=-1.0)
    x0 = np.random.uniform(size=1000)
    x1 = 2.0*x0 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])
    model = DummyBoostrapMixin()
    result = model.bootstrap(X, n_sampling=5)
    try:
        result.get_directed_acyclic_graph_counts(min_causal_effect=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError
