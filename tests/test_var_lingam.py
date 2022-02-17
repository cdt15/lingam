import os

import numpy as np
import pandas as pd

from lingam.direct_lingam import DirectLiNGAM
from lingam.var_lingam import VARLiNGAM


def generate_data(n=5, T=1000, random_state=None, initial_data=None):
    """
    Parameter
    ---------
    n : int
        number of variables
    T : int
        number of samples
    random_state : int
        seed for np.random.seed
    initial_data : list of np.ndarray
        dictionary of initial datas
    """

    T_spurious = 20
    expon = 1.5
    
    if initial_data is None:
        permutation = np.random.permutation(n)
        
        value = np.random.uniform(low=0.05, high=0.5, size=(n, n))
        sign = np.random.choice([-1, 1], size=(n, n))
        B0 = np.multiply(value, sign)
        
        B0 = np.multiply(B0, np.random.binomial(1, 0.4, size=(n, n)))
        B0 = np.tril(B0, k=-1)
        B0 = B0[permutation][:, permutation]

        value = np.random.uniform(low=0.05, high=0.5, size=(n, n))
        sign = np.random.choice([-1, 1], size=(n, n))
        B1 = np.multiply(value, sign)
        B1 = np.multiply(B1, np.random.binomial(1, 0.4, size=(n, n)))
        
        causal_order = np.empty(len(permutation))
        causal_order[permutation] = np.arange(len(permutation))
        causal_order = causal_order.astype(int)
    else:
        B0 = initial_data['B0']
        B1 = initial_data['B1']
        causal_order =initial_data['causal_order'] 
        
    M1 = np.dot(np.linalg.inv(np.eye(n) - B0), B1);

    ee = np.empty((n, T + T_spurious))
    for i in range(n):
        ee[i, :] = np.random.normal(size=(1, T + T_spurious));
        ee[i, :] = np.multiply(np.sign(ee[i, :]), abs(ee[i, :]) ** expon);
        ee[i, :] = ee[i, :] - np.mean(ee[i, :]);
        ee[i, :] = ee[i, :] / np.std(ee[i, :]);

    std_e = np.random.uniform(size=(n,)) + 0.5
    nn = np.dot(np.dot(np.linalg.inv(np.eye(n) - B0), np.diag(std_e)), ee);

    xx = np.zeros((n, T + T_spurious))
    xx[:, 0] = np.random.normal(size=(n, ));

    for t in range(1, T + T_spurious):
        xx[:, t] = np.dot(M1, xx[:, t - 1]) + nn[:, t];

    data = xx[:, T_spurious + 1 : T_spurious + T];
    
    return data.T, B0, B1, causal_order

def test_fit_success():
    X, B0, B1, causal_order = generate_data(n=3, T=100)

    # default
    model = VARLiNGAM()
    model.fit(X)
    co = model.causal_order_
    am = model.adjacency_matrices_
    resid = model.residuals_
    p_values = model.get_error_independence_p_values()

    # lags=2
    model = VARLiNGAM(lags=2)
    model.fit(X)

    # criterion='aic'
    model = VARLiNGAM(criterion='aic')
    model.fit(X)

    # prune=True
    model = VARLiNGAM(prune=True)
    model.fit(X)

    # ar_coefs
    model = VARLiNGAM(ar_coefs=[[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]])
    model.fit(X)

    # lingam_model
    m = DirectLiNGAM()
    model = VARLiNGAM(lingam_model=m)
    model.fit(X)

def test_fit_invalid():
    X, B0, B1, causal_order = generate_data(n=3, T=100)

    # invalid lingam_model
    try:
        model = VARLiNGAM(lingam_model=1)
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

def test_bootstrap_success():
    X, B0, B1, causal_order = generate_data(n=3, T=100)

    model = VARLiNGAM()
    result = model.bootstrap(X, n_sampling=3)

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

    # get_paths
    result.get_paths(1, 0, 1, 0)

def test_bootstrap_invalid():
    X, B0, B1, causal_order = generate_data(n=3, T=100)

    model = VARLiNGAM()
    result = model.bootstrap(X, n_sampling=3)

    result.adjacency_matrices_
    result.total_effects_

    # min_causal_effect
    try:
        result.get_paths(1, 0, 0, 1, -1)
    except ValueError:
        pass
    else:
        raise AssertionError

    # to_lag > from_lag
    try:
        result.get_paths(1, 0, 0, 1)
    except ValueError:
        pass
    else:
        raise AssertionError

    # same variable
    try:
        result.get_paths(0, 0, 1, 1)
    except ValueError:
        pass
    else:
        raise AssertionError

def test_estimate_total_effect_invalid():
    X, B0, B1, causal_order = generate_data(n=3, T=100)

    model = VARLiNGAM()
    model.fit(X)
    model._causal_order = [0, 1, 2]

    # warning
    model.estimate_total_effect(X, 2, 1)
