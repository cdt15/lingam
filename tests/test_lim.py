"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""

import numpy as np
# import scipy.optimize as sopt
# from scipy.special import expit as sigmoid
# import time
# import pandas as pd
# import networkx as nx
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.metrics import log_loss
# from Combining import likelihood
import random
# import itertools
# from itertools import product, combinations, permutations, chain
import sys
sys.path.append("..") 
from lingam import utils as ut
import lingam

def generate_data(n_features,n_samples,n_edges):

    graph_type, sem_type = 'ER', 'mixed_random_i_dis'
    B_true = ut.simulate_dag(n_features, n_edges, graph_type)
    W_true = ut.simulate_parameter(B_true)  # row to column

    no_dis = np.random.randint(1, n_features)  # number of discrete vars.
    print('There are %d discrete vars.' % (no_dis))
    nodes = [iii for iii in range(n_features)]
    dis_var = random.sample(nodes, no_dis)  # randomly select no_dis discrete variables
    dis_con = np.full((1, n_features), np.inf)
    for iii in range(n_features):
        if iii in dis_var:
            dis_con[0, iii] = 0  # 1:continuous;   0:discrete
        else:
            dis_con[0, iii] = 1
    data = ut.simulate_linear_mixed_sem(W_true, n_samples, sem_type, dis_con)

    return  data, dis_con, W_true


def test_fit_lina():
    i = 1
    ut.set_random_seed(i)
    n_features, n_samples = 5, 5000
    n_edges = np.random.randint(4,11)
    X, dis_con, W_true = generate_data(n_features, n_samples, n_edges)
    model = lingam.LiM()
    model.fit(X, dis_con, only_global=True)

    print('The estimated adjacency matrix is:\n', model.adjacency_matrix_)
    print('The true adjacency matrix is:\n', W_true)
    print('Done.')