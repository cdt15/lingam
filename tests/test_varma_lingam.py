import os

import numpy as np
import pandas as pd

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
        ext[i, :] = np.random.normal(size=(1, T));
        ext[i, :] = np.multiply(np.sign(ext[i, :]), abs(ext[i, :]) ** expon);
        ext[i, :] = ext[i, :] - np.mean(ext[i, :]);
        ext[i, :] = ext[i, :] / np.std(ext[i, :]);

    # observed signals y
    y = np.zeros((n, T))
    y[:, 0] = np.random.normal(loc=0.1, scale=1, size=(n, )) * np.random.choice([-1, 1], size=(n, ))
    for t in range(1, T):
        for i in causal_order:
            y[i, t] = np.dot(psi0[i, :], y[:, t]) + np.dot(psi1[i, :], y[:, t - 1]) + ext[i, t] + np.dot(omega1[i, :], ext[:, t - 1])


    return y[:, head:].T, psi0, psi1, omega1, causal_order

def test_fit_success():
    initial_data = {}
    initial_data['psi0'] = np.array([
        [0, 0.2669171, -0.16719712],
        [0, 0, 0],
        [0, -0.92769185, 0],
    ])
    initial_data['phi1'] = np.array([
        [-0.50941033, -0.01429937, 0.09002112],
        [0.09321691, -0.44028983, -0.05818995],
        [-0.12986617, -0.88781915, 0.21726865],
    ])
    initial_data['omega1'] = np.array([
        [0.02264769, 0.29487095, 0.29243977],
        [-0.15626269, 0.22860591, 0.11884103],
        [-0.09901518, 0.45200271, 0.05312345],
    ])
    initial_data['theta1'] = np.array([
        [-0.02674394, 0.31577469, 0.33371151],
        [-0.15626269, 0.22860591, 0.11884103],
        [0.04594844, 0.23992688, -0.05712441],
    ])
    initial_data['causal_order'] = [1, 2, 0]
    X, psi0, phi1, omega1, causal_order = generate_data(n=3, T=500, initial_data=initial_data)

    model = VARMALiNGAM(order=(1, 1), criterion=None)
    model.fit(X)

    # check the causal ordering
    co = model.causal_order_
    assert co.index(1) < co.index(2) < co.index(0)

    # check the adjacency matrix
    psi0 = model.adjacency_matrices_[0][0]
    assert psi0[0, 1] > 0.2 and psi0[0, 2] < -0.1 and psi0[2, 1] < -0.8

    psi0[0, 1] = 0.0
    psi0[0, 2] = 0.0
    psi0[2, 1] = 0.0
    assert np.sum(psi0) < 0.1
