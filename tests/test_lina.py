"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/site/sshimizu06/lingam
"""


import numpy as np
import scipy.linalg
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
from factor_analyzer import (ConfirmatoryFactorAnalyzer, ModelSpecificationParser)
import random
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import lingam
import lingam.utils as ut


def test_fit_lina():
    i = np.random.randint(1000)
    ut.set_random_seed(i)
    noise_ratios = 0.1
    n_features = 10
    n_samples, n_features_latent, n_edges, graph_type, sem_type = 1000, 5, 5, 'ER', 'laplace'
    B_true = ut.simulate_dag(n_features_latent, n_edges, graph_type)
    W_true = ut.simulate_parameter(B_true)  # row to column

    f = ut.simulate_linear_sem(W_true, n_samples, sem_type)
    f_nor = np.zeros([n_samples, n_features_latent])
    scale = np.zeros([1, n_features_latent])
    W_true_scale = np.zeros([n_features_latent, n_features_latent])
    for j in range(n_features_latent):
        scale[0, j] = np.std(f[:, j])
        f_nor[:, j] = f[:, j] / np.std(f[:, j])
        W_true_scale[:, j] = W_true[:, j] / scale[0, j]  # scaled W_true

    # generate noises ei of xi
    e = np.random.random([n_features, n_samples])
    for j in range(n_features):
        e[j, :] = e[j, :] - np.mean(e[j, :])
        e[j, :] = e[j, :] / np.std(e[j, :])

    G = np.zeros([n_features, n_features_latent])
    G[0, 0] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G[1, 0] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G[2, 1] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G[3, 1] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G[4, 2] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G[5, 2] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G[6, 3] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G[7, 3] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G[8, 4] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G[9, 4] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G_sign = np.sign(G)

    # normalize G
    G_nor = np.zeros([n_features, n_features_latent])
    for j in range(n_features):
        e[j, :] = e[j, :] / np.sqrt(np.square(np.sum(G[j, :])) + np.square(noise_ratios))
        G_nor[j, :] = G[j, :] / np.sqrt(np.square(np.sum(G[j, :])) + np.square(noise_ratios))

    X = G_nor @ f_nor.T + noise_ratios * e  # "e is small or n_features are large"
    X = X.T
    print(G_nor)

    model = lingam.LiNA()
    model.fit(X, G_sign, scale)

    print('The estimated adjacency matrix is:\n', model.adjacency_matrix_)
    print('The true adjacency matrix is:\n', W_true)


def test_fit_mdlina():
    i = np.random.randint(1000)
    ut.set_random_seed(i)
    n_features = 6
    noise_ratios = 0.1

    n_samples, n_features_latent, n_edges, graph_type, sem_type1, sem_type2 = \
        1000, 3, 3, 'ER', 'subGaussian', 'supGaussian'
    B_true = ut.simulate_dag(n_features_latent, n_edges, graph_type)  # skeleton btw. latent factors
    W_true = ut.simulate_parameter(B_true)  # causal effects matrix btw. latent factors

    # 1 domain
    f = ut.simulate_linear_sem(W_true, n_samples, sem_type1)
    f_nor1 = np.zeros([n_samples, n_features_latent])
    scale1 = np.zeros([1, n_features_latent])
    W_true_scale = np.zeros([n_features_latent, n_features_latent])
    for j in range(n_features_latent):
        scale1[0, j] = np.std(f[:, j])
        f_nor1[:, j] = f[:, j] / np.std(f[:, j])
        W_true_scale[:, j] = W_true[:, j] / scale1[0, j]
    e = np.random.random([n_features, n_samples])
    for j in range(n_features):
        e[j, :] = e[j, :] - np.mean(e[j, :])
        e[j, :] = e[j, :] / np.std(e[j, :])

    G1 = np.zeros([n_features, n_features_latent])
    G1[0, 0] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G1[1, 0] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G1[2, 1] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G1[3, 1] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G1[4, 2] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G1[5, 2] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G_sign1 = np.sign(G1)
    # normalize G
    G_nor1 = np.zeros([n_features, n_features_latent])
    for j in range(n_features):
        e[j, :] = e[j, :] / np.sqrt(np.square(np.sum(G1[j, :])) + np.square(noise_ratios))
        G_nor1[j, :] = G1[j, :] / np.sqrt(np.square(np.sum(G1[j, :])) + np.square(noise_ratios))
    X1 = G_nor1 @ f_nor1.T + noise_ratios * e  # "the noise ratio e is small or n_features is large"
    X1 = X1.T

    # 2 domain
    f2 = ut.simulate_linear_sem(W_true, n_samples, sem_type2)
    f_nor2 = np.zeros([n_samples, n_features_latent])
    scale2 = np.zeros([1, n_features_latent])
    W_true_scale = np.zeros([n_features_latent, n_features_latent])
    for j in range(n_features_latent):
        scale2[0, j] = np.std(f2[:, j])
        f_nor2[:, j] = f2[:, j] / np.std(f2[:, j])
        W_true_scale[:, j] = W_true[:, j] / scale2[0, j]
    e = np.random.random([n_features, n_samples])
    for j in range(n_features):
        e[j, :] = e[j, :] - np.mean(e[j, :])
        e[j, :] = e[j, :] / np.std(e[j, :])
    G2 = np.zeros([n_features, n_features_latent])
    G2[0, 0] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G2[1, 0] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G2[2, 1] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G2[3, 1] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G2[4, 2] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G2[5, 2] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G_sign2 = np.sign(G2)
    # normalize G
    G_nor2 = np.zeros([n_features, n_features_latent])
    for j in range(n_features):
        e[j, :] = e[j, :] / np.sqrt(np.square(np.sum(G2[j, :])) + np.square(noise_ratios))
        G_nor2[j, :] = G2[j, :] / np.sqrt(np.square(np.sum(G2[j, :])) + np.square(noise_ratios))
    X2 = G_nor2 @ f_nor2.T + noise_ratios * e
    X2 = X2.T

    # augment the data X
    XX = scipy.linalg.block_diag(X1, X2)
    G_sign = scipy.linalg.block_diag(G_sign1, G_sign2)
    scale = scipy.linalg.block_diag(scale1, scale2)
    print(G_sign)

    model = lingam.MDLiNA()
    model.fit(XX, G_sign, scale)

    print('The estimated adjacency matrix is:\n', model._adjacency_matrix)
    print('The true adjacency matrix is:\n', W_true)
