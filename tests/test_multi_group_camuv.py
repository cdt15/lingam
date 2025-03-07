import os

import numpy as np
import random
import lingam


def get_noise(n):
    noise = ((np.random.rand(1, n) - 0.5) * 5).reshape(n)
    mean = get_random_constant(0.0, 2.0)
    noise += mean
    return noise

def causal_func(cause):
    a = get_random_constant(-5.0, 5.0)
    b = get_random_constant(-1.0, 1.0)
    c = int(random.uniform(2, 3))
    return ((cause + a) ** (c)) + b

def get_random_constant(s, b):
    constant = random.uniform(-1.0, 1.0)
    if constant > 0:
        constant = random.uniform(s, b)
    else:
        constant = random.uniform(-b, -s)
    return constant

def create_data(n):
    causal_pairs = [[0, 1], [0, 3], [2, 4]]
    intermediate_pairs = [[2, 5]]
    confounder_pairs = [[3, 4]]

    n_variables = 6

    data = np.zeros((n, n_variables))  # observed data
    confounders = np.zeros(
        (n, len(confounder_pairs))
    )  # data of unobserced common causes

    # Adding external effects
    for i in range(n_variables):
        data[:, i] = get_noise(n)
    for i in range(len(confounder_pairs)):
        confounders[:, i] = get_noise(n)
        confounders[:, i] = confounders[:, i] / np.std(confounders[:, i])

    # Adding the effects of unobserved common causes
    for i, cpair in enumerate(confounder_pairs):
        cpair = list(cpair)
        cpair.sort()
        data[:, cpair[0]] += causal_func(confounders[:, i])
        data[:, cpair[1]] += causal_func(confounders[:, i])

    for i1 in range(n_variables)[0:n_variables]:
        data[:, i1] = data[:, i1] / np.std(data[:, i1])
        for i2 in range(n_variables)[i1 + 1 : n_variables + 1]:
            # Adding direct effects between observed variables
            if [i1, i2] in causal_pairs:
                data[:, i2] += causal_func(data[:, i1])
            # Adding undirected effects between observed variables mediated through unobserved variables
            if [i1, i2] in intermediate_pairs:
                interm = causal_func(data[:, i1]) + get_noise(n)
                interm = interm / np.std(interm)
                data[:, i2] += causal_func(interm)

    return data

def test_fit_success():
    X1 = create_data(200)
    X2 = create_data(200)
    X_list = [X1, X2]
    model = lingam.MultiGroupCAMUV()
    model.fit(X_list)
    print(model.adjacency_matrix_)

    # f-correlation
    model = lingam.MultiGroupCAMUV(independence="fcorr", ind_corr=0.5)
    model.fit(X_list)

    # prior_knowledge
    model = lingam.MultiGroupCAMUV(prior_knowledge=[(1, 0)])
    model.fit(X_list)

def test_fit_invalid():
    try:
        X1 = create_data(200)
        X2 = create_data(200)
        X_list = [X1, X2]
        model = lingam.MultiGroupCAMUV(alpha=-1)
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    try:
        X1 = create_data(200)
        X2 = create_data(200)
        X_list = [X1, X2]
        model = lingam.MultiGroupCAMUV(num_explanatory_vals=-1)
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Invalid value: independence
    try:
        model = lingam.MultiGroupCAMUV(independence="lingam")
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError

    try:
        X = create_data(200)
        model = lingam.MultiGroupCAMUV(ind_corr=-1.0)
        model.fit(X_list)
    except ValueError:
        pass
    else:
        raise AssertionError
