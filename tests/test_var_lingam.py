import os

import numpy as np
import pandas as pd

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
    initial_data = {}
    initial_data['B0'] = [
        [0, 0.48364824, 0],
        [0, 0, -0.24064466],
        [0, 0, 0],
    ]
    initial_data['B1'] = [
        [-0.35549579, -0.37428469, -0.31190891],
        [0, 0, 0.09765842],
        [0, -0.13384955, -0.38161318],
    ]
    initial_data['causal_order'] = [2, 1, 0]
    
    X, B0, B1, causal_order = generate_data(n=3, T=1000, initial_data=initial_data)

    model = VARLiNGAM(lags=1, criterion=None)
    model.fit(X)

    # check the causal ordering
    co = model.causal_order_
    assert co.index(2) < co.index(1) < co.index(0)

    # check the adjacency matrix
    b0 = model.adjacency_matrices_[0]
    assert b0[0, 1] > 0.4 and b0[1, 2] < -0.1

    b0[0, 1] = 0.0
    b0[1, 2] = 0.0
    assert np.sum(b0) < 0.1

    b1 = model.adjacency_matrices_[1]
    assert b1[0, 0] < -0.3 and b1[0, 1] < -0.3 and b1[0, 2] < -0.3 \
       and b1[1, 2] > 0.05 and b1[2, 1] < -0.1 and b1[2, 2] < -0.3

    b1[0, 0] = 0.0
    b1[0, 1] = 0.0
    b1[0, 2] = 0.0
    b1[1, 2] = 0.0
    b1[2, 1] = 0.0
    b1[2, 2] = 0.0
    assert np.sum(b1) < 0.1
