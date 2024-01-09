import os

import numpy as np
import pandas as pd

from lingam.rcd import RCD

def test_fit_success():
    # causal direction: x5 --> x0, x3 --> x1 --> x2, x4 <-- x6
    get_external_effect = lambda n: np.random.normal(0.0, 0.5, n) ** 3
    n_samples = 100
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

    # default
    model = RCD()
    model.fit(X)

    ans = model.ancestors_list_
    am = model.adjacency_matrix_
    p_values = model.get_error_independence_p_values(X)

    # max_explanatory_num=3
    model = RCD(max_explanatory_num=3)
    model.fit(X)

    # max_explanatory_num=1
    model = RCD(max_explanatory_num=1)
    model.fit(X)

    # cor_alpha=0.1
    model = RCD(cor_alpha=0.1)
    model.fit(X)

    # ind_alpha=0.1
    model = RCD(ind_alpha=0.1)
    model.fit(X)

    # shapiro_alpha=0.1
    model = RCD(shapiro_alpha=0.1)
    model.fit(X)

    # shapiro_alpha=0.1
    model = RCD(shapiro_alpha=0.0)
    model.fit(X)

    # MLHSICR=True
    model = RCD(MLHSICR=True)
    model.fit(X)

    # bw_method='scott'
    model = RCD(bw_method='scott')
    model.fit(X)

    # bw_method='silverman'
    model = RCD(bw_method='silverman')
    model.fit(X)

    # no latent confounders
    get_external_effect = lambda n: np.random.normal(0.0, 0.5, n) ** 3
    n_samples = 100
    x0 = get_external_effect(n_samples)
    x3 = get_external_effect(n_samples)
    x1 = 1.0*x0 + 1.0*x3 + get_external_effect(n_samples)
    x2 = 0.8*x1 + get_external_effect(n_samples)
    x4 = 1.0*x1 + get_external_effect(n_samples)
    X = pd.DataFrame(np.array([x0, x1, x2, x3, x4]).T, columns=['x0', 'x1', 'x2', 'x3', 'x4'])
    model = RCD()
    model.fit(X)
    p_values = model.get_error_independence_p_values(X)

    # causal direction: x3-->x0, x3-->x1, x0,x1-->x2
    get_external_effect = lambda n: np.random.normal(0.0, 0.5, n) ** 3
    n_samples = 100
    x3 = get_external_effect(n_samples)
    x0 = 0.5*x3 + get_external_effect(n_samples)
    x1 = 0.5*x3 + get_external_effect(n_samples)
    x2 = 1.0*x0 + 1.0*x1 + get_external_effect(n_samples)
    X = pd.DataFrame(np.array([x0, x1, x2]).T, columns=['x0', 'x1', 'x2'])
    model = RCD()
    model.fit(X)
    p_values = model.get_error_independence_p_values(X)

    # f-correlation
    model = RCD(independence="fcorr")
    model.fit(X)



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
    x0 = np.random.uniform(size=100)
    x1 = 2.0*x0 + np.random.uniform(size=100)
    X = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])
    X.iloc[10, 0] = np.nan
    try:
        model = RCD()
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Include infinite values
    x0 = np.random.uniform(size=100)
    x1 = 2.0*x0 + np.random.uniform(size=100)
    X = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])
    X.iloc[10, 0] = np.inf
    try:
        model = RCD()
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid value: max_explanatory_num
    x0 = np.random.uniform(size=100)
    x1 = 2.0*x0 + np.random.uniform(size=100)
    X = pd.DataFrame(np.array([x0, x1]).T, columns=['x0', 'x1'])
    try:
        model = RCD(max_explanatory_num=-1)
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid value: cor_alpha
    try:
        model = RCD(cor_alpha=-1)
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid value: ind_alpha
    try:
        model = RCD(ind_alpha=-1)
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid value: shapiro_alpha
    try:
        model = RCD(shapiro_alpha=-1)
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid value: bw_method
    try:
        model = RCD(bw_method='X')
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid value: independence
    try:
        model = RCD(independence="lingam")
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid value: ind_corr
    try:
        model = RCD(ind_corr=-0.5)
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

def test_bootstrap_success():
    get_external_effect = lambda n: np.random.normal(0.0, 0.5, n) ** 3
    n_samples = 100
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

    model = RCD()
    result = model.bootstrap(X, n_sampling=20)

    result.adjacency_matrices_
    result.total_effects_

    # No argument
    cdc = result.get_causal_direction_counts()

    # n_directions=2
    cdc = result.get_causal_direction_counts(n_directions=2)

    # min_causal_effect=0.2
    cdc = result.get_causal_direction_counts(min_causal_effect=0.2)

    # split_by_causal_effect_sign=True
    cdc = result.get_causal_direction_counts(split_by_causal_effect_sign=True)

    # No argument
    dagc = result.get_directed_acyclic_graph_counts()

    # n_dags=2
    dagc = result.get_directed_acyclic_graph_counts(n_dags=2)

    # min_causal_effect=0.6
    dagc = result.get_directed_acyclic_graph_counts(min_causal_effect=0.6)

    # split_by_causal_effect_sign=True
    dagc = result.get_directed_acyclic_graph_counts(split_by_causal_effect_sign=True)

    # get_probabilities
    probs = result.get_probabilities()

    # get_probabilities
    probs = result.get_probabilities(min_causal_effect=0.6)

    # get_total_causal_effects
    ce = result.get_total_causal_effects()

    # get_total_causal_effects
    ce = result.get_total_causal_effects(min_causal_effect=0.6)

    # MLHSICR=True
    model = RCD(MLHSICR=True)
    result = model.bootstrap(X, n_sampling=20)

    # no latent confounders
    get_external_effect = lambda n: np.random.normal(0.0, 0.5, n) ** 3
    n_samples = 100
    x0 = get_external_effect(n_samples)
    x3 = get_external_effect(n_samples)
    x1 = 1.0*x0 + 1.0*x3 + get_external_effect(n_samples)
    x2 = 0.8*x1 + get_external_effect(n_samples)
    x4 = 1.0*x1 + get_external_effect(n_samples)
    X = pd.DataFrame(np.array([x0, x1, x2, x3, x4]).T, columns=['x0', 'x1', 'x2', 'x3', 'x4'])

    model = RCD()
    result = model.bootstrap(X, n_sampling=20)

    # causal direction: x3-->x0, x3-->x1, x0,x1-->x2
    get_external_effect = lambda n: np.random.normal(0.0, 0.5, n) ** 3
    n_samples = 100
    x3 = get_external_effect(n_samples)
    x0 = 0.5*x3 + get_external_effect(n_samples)
    x1 = 0.5*x3 + get_external_effect(n_samples)
    x2 = 1.0*x0 + 1.0*x1 + get_external_effect(n_samples)
    X = pd.DataFrame(np.array([x0, x1, x2]).T, columns=['x0', 'x1', 'x2'])

    model = RCD()
    result = model.bootstrap(X, n_sampling=20)

def test_bootstrap_invalid():
    get_external_effect = lambda n: np.random.normal(0.0, 0.5, n) ** 3
    n_samples = 100
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

    model = RCD()

    # Invalid value: n_sampling=-1
    try:
        result = model.bootstrap(X, n_sampling=-1)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid value: n_sampling='3'
    try:
        result = model.bootstrap(X, n_sampling='3')
    except ValueError:
        pass
    else:
        raise AssertionError



def test_estimate_total_effect_invalid():
    get_external_effect = lambda n: np.random.normal(0.0, 0.5, n) ** 3
    n_samples = 100
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

    model = RCD()
    model.fit(X)
    model._ancestors_list[1] = [2]
    model._ancestors_list[2] = [3, 4]
    model._adjacency_matrix = [[0, np.nan], [0, np.nan], [0, np.nan]]

    # warning
    model.estimate_total_effect(X, 1, 2)
    model.estimate_total_effect(X, 2, 1)
