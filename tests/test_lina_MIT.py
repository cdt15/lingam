"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""


import numpy as np
import sys
sys.path.append('D:/Codes/Git/lingam')
import lingam
import os

def test_fit_lina():
    # load data
    DATA_DIR_PATH = os.path.dirname(__file__)
    X = np.loadtxt(f"{DATA_DIR_PATH}/test_lina_data.csv", delimiter=",")
    W_true = np.array(
        [[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 1.23784047,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        , -1.49650548,  0.        ],
       [-1.05331666, -0.52543143,  0.        ,  0.        ,  0.50714686],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])
    G_sign = np.array(
        [[ 1.,  0.,  0.,  0.,  0.],
       [-1.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  0.],
       [ 0., -1.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0., -1.,  0.],
       [ 0.,  0.,  0., -1.,  0.],
       [ 0.,  0.,  0.,  0., -1.],
       [ 0.,  0.,  0.,  0., -1.]])
    scale = np.array([[4.62688314, 1.84996207, 1.36308856, 2.39533958, 1.95656385]])
    
    model = lingam.LiNA()
    model.fit(X, G_sign, scale)

    print('The estimated adjacency matrix is:\n', model.adjacency_matrix_)
    print('The true adjacency matrix is:\n', W_true)


def test_fit_mdlina():
    # load data
    DATA_DIR_PATH = os.path.dirname(__file__)
    XX = np.loadtxt(f"{DATA_DIR_PATH}/test_mdlina_data.csv", delimiter=",")
    W_true = np.array(
        [[ 0.        ,  1.02343092, -1.70436068],
       [ 0.        ,  0.        , -1.47895291],
       [ 0.        ,  0.        ,  0.        ]])
    G_sign = np.array(
        [[ 1.,  0.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.,  0.],
       [ 0., -1.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  0.,  0., -1.],
       [ 0.,  0.,  0.,  0.,  0.,  1.]]
    )
    scale = np.array([[1.        , 1.42970805, 3.66739664, 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 1.        , 1.45710481,
        3.70389115]])

    model = lingam.MDLiNA()
    model.fit(XX, G_sign, scale)

    print('The estimated adjacency matrix is:\n', model._adjacency_matrix)
    print('The true adjacency matrix is:\n', W_true)
