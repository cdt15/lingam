import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
import graphviz
import lingam
from lingam.utils import make_dot, get_cuda_version

def main():
    cuda = get_cuda_version()

    if cuda:
        print([np.__version__, pd.__version__, graphviz.__version__, lingam.__version__])

        np.set_printoptions(precision=3, suppress=True)
        np.random.seed(100)

        # ## Test data
        # We create test data consisting of 6 variables.

        x3 = np.random.uniform(size=1000)
        x0 = 3.0*x3 + np.random.uniform(size=1000)
        x2 = 6.0*x3 + np.random.uniform(size=1000)
        x1 = 3.0*x0 + 2.0*x2 + np.random.uniform(size=1000)
        x5 = 4.0*x0 + np.random.uniform(size=1000)
        x4 = 8.0*x0 - 1.0*x2 + np.random.uniform(size=1000)
        X = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T ,columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])
        X.head()

        # ## Causal Discovery
        # To run causal discovery, we create a `DirectLiNGAM` object and call the `fit` method.
        # We use the pwling_fast measure which uses a CUDA accelerated implementation of pwling

        model = lingam.DirectLiNGAM(measure='pwling')
        model.fit(X)

        print(model.causal_order_)
        print(model.adjacency_matrix_)

        m = model.adjacency_matrix_

        model = lingam.DirectLiNGAM(measure='pwling_fast')
        model.fit(X)

        assert np.allclose(model.adjacency_matrix_, m)

        # ## Independence between error variables
        # To check if the LiNGAM assumption is broken, we can get p-values of independence between error variables. The value in the i-th row and j-th column of the obtained matrix shows the p-value of the independence of the error variables $e_i$ and $e_j$.

        p_values = model.get_error_independence_p_values(X)
        print(p_values)

if __name__ == "__main__":
    main()
