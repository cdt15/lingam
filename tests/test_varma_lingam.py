import os

import numpy as np
import pandas as pd

from lingam.direct_lingam import DirectLiNGAM
from lingam.varma_lingam import VARMALiNGAM

def randnetbalanced(dims, samples, indegree, parminmax, errminmax):
    """
    この関数は以前頂いたmatlabのスクリプトを移植したものですのでご確認不要です。
    create a more balanced random network

    Parameter
    ---------
    dims : int
        number of variables
    samples : int
        number of samples
    indegree : int or float('inf')
        number of parents of each node (float('inf') = fully connected)
    parminmax : dictionary
        standard deviation owing to parents 
    errminmax : dictionary
        standard deviation owing to error variable

    Return
    ------
    B : array, shape (dims, dims)
        the strictly lower triangular network matrix
    errstd : array, shape (dims, 1)
        the vector of error (disturbance) standard deviations
    """

    # First, generate errstd
    errstd = np.random.uniform(low=errminmax['min'], high=errminmax['max'], size=(dims, 1))

    # Initializations
    X = np.empty(shape=[dims, samples])
    B = np.zeros([dims, dims])

    # Go trough each node in turn
    for i in range(dims):

        # If indegree is finite, randomly pick that many parents,
        # else, all previous variables are parents
        if indegree == float('inf'):
            if i <= indegree:
                par = np.arange(i)
            else:
                par = np.random.permutation(i)[:indegree]
        else:
            par = np.arange(i)

        if len(par) == 0:
            # if node has no parents
            # Increase errstd to get it to roughly same variance
            parent_std = np.random.uniform(low=parminmax['min'], high=parminmax['max'])
            errstd[i] = np.sqrt(errstd[i]**2 + parent_std**2)

            # Set data matrix to empty
            X[i] = np.zeros(samples)
        else:
            # If node has parents, do the following
            w = np.random.normal(size=[1, len(par)])

            # Randomly pick weights
            wfull = np.zeros([1, i])
            wfull[0, par] = w

            # Calculate contribution of parents
            X[i] = np.dot(wfull, X[:i, :])

            # Randomly select a 'parents std' 
            parstd = np.random.uniform(low=parminmax['min'], high=parminmax['max'])

            # Scale w so that the combination of parents has 'parstd' std
            scaling = parstd / np.sqrt(np.mean(X[i] ** 2))
            w = w * scaling

            # Recalculate contribution of parents
            wfull = np.zeros([1, i])
            wfull[0, par] = w
            X[i] = np.dot(wfull, X[:i, :])

            # Fill in B
            B[i, par] = w

        # Update data matrix
        X[i] = X[i] + np.random.normal(size=samples) * errstd[i]

    return B, errstd

def generate_data(n=5, T=800, initial_data=None):
    head = 100
    T = T + head
    
    if initial_data is None:
        # psi0
        indegree = float('inf')
        psi0, _ = randnetbalanced(n, n, indegree, {'min':0.05, 'max':0.5}, {'min':0.05, 'max':0.5})
        permutation = np.random.permutation(n)
        psi0 = psi0[permutation][:, permutation]

        # causal order
        causal_order = np.empty(len(permutation))
        causal_order[permutation] = np.arange(len(permutation))
        causal_order = causal_order.astype(int)

        # phi1
        value = np.random.uniform(low=0.01, high=0.5, size=(n, n))
        sign = np.random.choice([-1, 1], size=(n, n))
        phi1 = np.multiply(value, sign)

        # theta1
        value = np.random.uniform(low=0.01, high=0.5, size=(n, n))
        sign = np.random.choice([-1, 1], size=(n, n))
        theta1 = np.multiply(value, sign)
    else:
        psi0 = initial_data['psi0']
        phi1 = initial_data['phi1']
        theta1 = initial_data['theta1']
        causal_order = initial_data['causal_order']

    # psi1, omega1
    psi1 = np.dot(np.eye(n) - psi0, phi1)
    omega1 = np.dot(np.eye(n) - psi0, theta1, np.linalg.inv(np.eye(n) - psi0))
                    
    # external influence
    expon = 0.1
    ext = np.empty((n, T))
    for i in range(n):
        ext[i, :] = np.random.normal(size=(1, T))
        ext[i, :] = np.multiply(np.sign(ext[i, :]), abs(ext[i, :]) ** expon)
        ext[i, :] = ext[i, :] - np.mean(ext[i, :])
        ext[i, :] = ext[i, :] / np.std(ext[i, :])

    # observed signals y
    y = np.zeros((n, T))
    y[:, 0] = np.random.normal(loc=0.1, scale=1, size=(n, )) * np.random.choice([-1, 1], size=(n, ))
    for t in range(1, T):
        for i in causal_order:
            y[i, t] = np.dot(psi0[i, :], y[:, t]) + np.dot(psi1[i, :], y[:, t - 1]) + ext[i, t] + np.dot(omega1[i, :], ext[:, t - 1])


    return y[:, head:].T, psi0, psi1, omega1, causal_order

def test_fit_success():
    X, psi0, phi1, omega1, causal_order = generate_data(n=3, T=30)

    # default
    model = VARMALiNGAM()
    model.fit(X)
    co = model.causal_order_
    am = model.adjacency_matrices_
    resid = model.residuals_
    p_values = model.get_error_independence_p_values()

    print('_ma_coefs:\n', model._ma_coefs)

    # order=(2, 1)
    model = VARMALiNGAM(order=(2, 2))
    model.fit(X)

    # criterion='aic'
    model = VARMALiNGAM(criterion='aic')
    model.fit(X)

    # prune=True
    model = VARMALiNGAM(prune=True)
    model.fit(X)

    model = VARMALiNGAM(prune=True, criterion=None, order=(2, 2))
    model.fit(X)

    # ar_coefs
    model = VARMALiNGAM(ar_coefs=[[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]])
    model.fit(X)

    # ma_coefs
    model = VARMALiNGAM(ma_coefs=[[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]])
    model.fit(X)

    # lingam_model
    m = DirectLiNGAM()
    model = VARMALiNGAM(lingam_model=m)
    model.fit(X)

def test_fit_invalid():
    X, psi0, phi1, omega1, causal_order = generate_data(n=3, T=30)

    # invalid lingam_model
    try:
        model = VARMALiNGAM(lingam_model=1)
        model.fit(X)
    except ValueError:
        pass
    else:
        raise AssertionError

def test_bootstrap_success():
    X, psi0, phi1, omega1, causal_order = generate_data(n=3, T=30)

    model = VARMALiNGAM()
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
    X, psi0, phi1, omega1, causal_order = generate_data(n=3, T=100)

    model = VARMALiNGAM()
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
    X, psi0, phi1, omega1, causal_order = generate_data(n=3, T=30)

    model = VARMALiNGAM()
    model.fit(X)
    model._causal_order = [0, 1, 2]

    # warning
    model.estimate_total_effect(X, model.residuals_, 2, 1)
