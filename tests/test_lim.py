"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""

import os

import numpy as np
import sys
sys.path.append('D:/Codes/Git/lingam')
import lingam

DATA_DIR_PATH = os.path.dirname(__file__)


def test_fit_lim():
    X = np.loadtxt(f"{DATA_DIR_PATH}/test_lim_data.csv", delimiter=",")
    dis_con = np.array([[0.0, 0.0, 0.0, 1.0, 1.0]])
    W_true = np.array(
        [
            [0.0, 1.09482609, -1.29270764, 0.0, -0.84424137],
            [0.0, 0.0, 0.80393307, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.70346053, 0.0, 1.90912441, 0.0, 1.94441713],
            [0.0, 0.0, -0.63152585, 0.0, 0.0],
        ]
    )
    model = lingam.LiM()
    model.fit(X, dis_con, only_global=False, is_poisson=True)

    print("The estimated adjacency matrix is:\n", model.adjacency_matrix_)
    print("The true adjacency matrix is:\n", W_true)
    print("Done.")

    model = lingam.LiM()
    model.fit(X, dis_con, only_global=False, is_poisson=False)
    # model.fit(X, dis_con, only_global=False)

    print("The estimated adjacency matrix is:\n", model.adjacency_matrix_)
    print("The true adjacency matrix is:\n", W_true)
    print("Done.")
