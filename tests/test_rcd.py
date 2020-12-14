import os

import numpy as np
import pandas as pd

from lingam.rcd import RCD

def test_fit_success():
    # causal direction: x5 --> x0, x3 --> x1 --> x2, x4 <-- x6
    get_external_effect = lambda n: np.random.normal(0.0, 0.5, n) ** 3
    n_samples = 1000
    x5 = get_external_effect(n_samples)
    x6 = get_external_effect(n_samples)
    x0 = 0.6*x5 + get_external_effect(n_samples)
    x3 = 0.5*x5 + get_external_effect(n_samples)
    x1 = 1.0*x0 + 1.0*x3 + get_external_effect(n_samples)
    x2 = 0.8*x1 - 0.6*x6 + get_external_effect(n_samples)
    x4 = 1.0*x1 - 0.5*x6 + get_external_effect(n_samples)
    # x5 and x6 are latent confounders
    X = pd.DataFrame(np.array([x0, x1, x2, x3, x4]).T,
                     columns=['x0', 'x1', 'x2', 'x3', 'x4'])

    model = RCD(MLHSICR=True)
    model.fit(X)

    # check the ancestors list
    ans = model.ancestors_list_
    assert set([0, 3]) == ans[1]
    assert set([0, 1, 3]) == ans[2]
    assert set([0, 1, 3]) == ans[4]

    # check the adjacency matrix
    am = model.adjacency_matrix_
    assert am[1, 0] > 0
    assert am[1, 3] > 0

    am[1, 0] = 0.0
    am[1, 3] = 0.0
    am[2, 1] = 0.0
    am[4, 1] = 0.0
    assert np.sum(np.nan_to_num(am)) < 0.1

def test_fit_invalid_data():
    # Not array data
    X = 1
    try:
        model = RCD()
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
        model = RCD()
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
        model = RCD()
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
        model = RCD()
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

if __name__ == "__main__":
    test_fit_success()
    test_fit_invalid_data()
