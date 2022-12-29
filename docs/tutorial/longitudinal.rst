
Longitudinal LiNGAM
===================

Model
-------------------
This method [2]_ performs causal discovery on paired samples based on longitudinal data that collects samples over time. 
Their algorithm can analyze causal structures, including topological causal orders, that may change over time.
Similarly to the basic LiNGAM model [1]_, this method makes the following assumptions:

#. Linearity
#. Non-Gaussian continuous error variables (except at most one)
#. Acyclicity
#. No hidden common causes

Denote observed variables and error variables of $m$-the sample at time point $t$ 
by $x_i^m (t)$ and $e_i^m (t)$. 
Collect them in vectors $ x^m (t) $ and $ e^m (t) $, respectivelly. 
Further, collect them in matrices $ X(t) $ and $ E(t) $. 

 and coefficients or connection strengths $b_{ij}^{(g)}$ ( $i,j=1, ..., p, g=1, ..., G$ ). 
Collect them in vectors $x$ and $e^{(g)}$ and a matrix $B^{(g)}$, respectivelly. 
Due to the assumptions of acyclicity and common topological causal orders, the adjacency matrix $B^{(g)}$ ( $g=1, ..., G$ ) 
can be permuted to be strictly lower-triangular by a simultaneous row and column permutation common to the groups. 
The error variables $e_i^{(g)}$ ( $i=1, ..., p$, $g=1, ..., G$ ) are independent due to the assumption of no hidden common causes. 

Then, mathematically, the model for observed variable vector $x^{(g)}$ is written as 

$$ x^{(g)} = B^{(g)}x^{(g)} + e^{(g)}. $$

References

    .. [1] S. Shimizu, P. O. Hoyer, A. Hyvärinen, and A. J. Kerminen.
       A linear non-gaussian acyclic model for causal discovery.
       Journal of Machine Learning Research, 7:2003-2030, 2006.
    .. [2] K. Kadowaki, S. Shimizu, and T. Washio. Estimation of causal structures in longitudinal data using non-Gaussianity. 
       In Proc. 23rd IEEE International Workshop on Machine Learning for Signal Processing (MLSP2013), pp. 1--6, Southampton, United Kingdom, 2013.


Import and settings
-------------------

In this example, we need to import ``numpy``, ``pandas``, and
``graphviz`` in addition to ``lingam``.

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import graphviz
    import lingam
    from lingam.utils import print_causal_directions, print_dagc, make_dot
    
    import warnings
    warnings.filterwarnings('ignore')
    
    print([np.__version__, pd.__version__, graphviz.__version__, lingam.__version__])
    
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(0)


.. parsed-literal::

    ['1.16.2', '0.24.2', '0.11.1', '1.5.2']
    

Test data
---------

We create test data consisting of 5 variables. The causal model at each
timepoint is as follows.

.. code-block:: python

    # setting
    n_features = 5
    n_samples = 200
    n_lags = 1
    n_timepoints = 3
    
    causal_orders = []
    B_t_true = np.empty((n_timepoints, n_features, n_features))
    B_tau_true = np.empty((n_timepoints, n_lags, n_features, n_features))
    X_t = np.empty((n_timepoints, n_samples, n_features))

.. code-block:: python

    # B(0,0)
    B_t_true[0] = np.array([[0.0, 0.5,-0.3, 0.0, 0.0],
                            [0.0, 0.0,-0.3, 0.4, 0.0],
                            [0.0, 0.0, 0.0, 0.3, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.1,-0.7, 0.0, 0.0, 0.0]])
    causal_orders.append([3, 2, 1, 0, 4])
    make_dot(B_t_true[0], labels=[f'x{i}(0)' for i in range(5)])




.. image:: ../image/longitudinal_dag1.svg



.. code-block:: python

    # B(1,1)
    B_t_true[1] = np.array([[0.0, 0.2,-0.1, 0.0,-0.5],
                            [0.0, 0.0, 0.0, 0.4, 0.0],
                            [0.0, 0.3, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0,-0.4, 0.0, 0.0, 0.0]])
    causal_orders.append([3, 1, 2, 4, 0])
    make_dot(B_t_true[1], labels=[f'x{i}(1)' for i in range(5)])




.. image:: ../image/longitudinal_dag2.svg



.. code-block:: python

    # B(2,2)
    B_t_true[2] = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0,-0.7, 0.0, 0.5],
                            [0.2, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0,-0.4, 0.0, 0.0],
                            [0.3, 0.0, 0.0, 0.0, 0.0]])
    causal_orders.append([0, 2, 4, 3, 1])
    make_dot(B_t_true[2], labels=[f'x{i}(2)' for i in range(5)])




.. image:: ../image/longitudinal_dag3.svg



.. code-block:: python

    # create B(t,t-τ) and X
    for t in range(n_timepoints):
        # external influence
        expon = 0.1
        ext = np.empty((n_features, n_samples))
        for i in range(n_features):
            ext[i, :] = np.random.normal(size=(1, n_samples));
            ext[i, :] = np.multiply(np.sign(ext[i, :]), abs(ext[i, :]) ** expon);
            ext[i, :] = ext[i, :] - np.mean(ext[i, :]);
            ext[i, :] = ext[i, :] / np.std(ext[i, :]);
    
        # create B(t,t-τ)
        for tau in range(n_lags):
            value = np.random.uniform(low=0.01, high=0.5, size=(n_features, n_features))
            sign = np.random.choice([-1, 1], size=(n_features, n_features))
            B_tau_true[t, tau] = np.multiply(value, sign)
    
        # create X(t)
        X = np.zeros((n_features, n_samples))
        for co in causal_orders[t]:
            X[co] = np.dot(B_t_true[t][co, :], X) + ext[co]
            if t > 0:
                for tau in range(n_lags):
                    X[co] = X[co] + np.dot(B_tau_true[t, tau][co, :], X_t[t-(tau+1)].T)
        
        X_t[t] = X.T

Causal Discovery
----------------

To run causal discovery, we create a :class:`~lingam.LongitudinalLiNGAM` object by specifying the ``n_lags`` parameter. Then, we call the :func:`~lingam.LongitudinalLiNGAM.fit` method.

.. code-block:: python

    model = lingam.LongitudinalLiNGAM(n_lags=n_lags)
    model = model.fit(X_t)

Using the :attr:`~lingam.LongitudinalLiNGAM.causal_orders_` property, we can see the causal ordering in time-points as a result of the causal discovery. All elements are nan because the causal order of B(t,t) at t=0 is not calculated. So access to the time points above t=1.

.. code-block:: python

    print(model.causal_orders_[0]) # nan at t=0
    print(model.causal_orders_[1])
    print(model.causal_orders_[2])


.. parsed-literal::

    [nan, nan, nan, nan, nan]
    [3, 1, 2, 4, 0]
    [0, 4, 2, 3, 1]
    

Also, using the :attr:`~lingam.LongitudinalLiNGAM.adjacency_matrices_` property, we can see the adjacency matrix as a result of the causal discovery. As with the causal order, all elements are nan because the B(t,t) and B(t,t-τ) at t=0 is not calculated. So access to the time points above t=1. Also, if we run causal discovery with n_lags=2, B(t,t-τ) at t=1 is also not computed, so all the elements are nan.

.. code-block:: python

    t = 0 # nan at t=0
    print('B(0,0):')
    print(model.adjacency_matrices_[t, 0])
    print('B(0,-1):')
    print(model.adjacency_matrices_[t, 1])
    
    t = 1
    print('B(1,1):')
    print(model.adjacency_matrices_[t, 0])
    print('B(1,0):')
    print(model.adjacency_matrices_[t, 1])
    
    t = 2
    print('B(2,2):')
    print(model.adjacency_matrices_[t, 0])
    print('B(2,1):')
    print(model.adjacency_matrices_[t, 1])


.. parsed-literal::

    B(0,0):
    [[nan nan nan nan nan]
     [nan nan nan nan nan]
     [nan nan nan nan nan]
     [nan nan nan nan nan]
     [nan nan nan nan nan]]
    B(0,-1):
    [[nan nan nan nan nan]
     [nan nan nan nan nan]
     [nan nan nan nan nan]
     [nan nan nan nan nan]
     [nan nan nan nan nan]]
    B(1,1):
    [[ 0.     0.099  0.     0.    -0.52 ]
     [ 0.     0.     0.     0.398  0.   ]
     [ 0.     0.384  0.    -0.162  0.   ]
     [ 0.     0.     0.     0.     0.   ]
     [ 0.    -0.249 -0.074  0.     0.   ]]
    B(1,0):
    [[ 0.025  0.116 -0.202  0.054 -0.216]
     [ 0.139 -0.211 -0.43   0.558  0.051]
     [-0.135  0.178  0.421  0.173  0.031]
     [ 0.384 -0.083 -0.495 -0.072 -0.323]
     [-0.206 -0.354 -0.199 -0.293  0.468]]
    B(2,2):
    [[ 0.     0.     0.     0.     0.   ]
     [ 0.     0.    -0.67   0.     0.46 ]
     [ 0.187  0.     0.     0.     0.   ]
     [ 0.     0.    -0.341  0.     0.   ]
     [ 0.25   0.     0.     0.     0.   ]]
    B(2,1):
    [[ 0.194  0.2    0.031 -0.473 -0.002]
     [-0.384 -0.037  0.158  0.255  0.095]
     [ 0.126  0.275 -0.048  0.502 -0.019]
     [ 0.238 -0.469  0.475 -0.029 -0.176]
     [-0.177  0.309 -0.112  0.295 -0.273]]
    

.. code-block:: python

    for t in range(1, n_timepoints):
        B_t, B_tau = model.adjacency_matrices_[t]
        plt.figure(figsize=(7, 3))
    
        plt.subplot(1,2,1)
        plt.plot([-1, 1],[-1, 1], marker="", color="blue", label="support")
        plt.scatter(B_t_true[t], B_t, facecolors='none', edgecolors='black')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.xlabel('True')
        plt.ylabel('Estimated')
        plt.title(f'B({t},{t})')
    
        plt.subplot(1,2,2)
        plt.plot([-1, 1],[-1, 1], marker="", color="blue", label="support")
        plt.scatter(B_tau_true[t], B_tau, facecolors='none', edgecolors='black')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.xlabel('True')
        plt.ylabel('Estimated')
        plt.title(f'B({t},{t-1})')
    
        plt.tight_layout()
        plt.show()



.. image:: ../image/longitudinal_scatter1.png



.. image:: ../image/longitudinal_scatter2.png


Independence between error variables
------------------------------------

To check if the LiNGAM assumption is broken, we can get p-values of
independence between error variables. The value in the i-th row and j-th
column of the obtained matrix shows the p-value of the independence of
the error variables :math:`e_i` and :math:`e_j`.

.. code-block:: python

    p_values_list = model.get_error_independence_p_values()

.. code-block:: python

    t = 1
    print(p_values_list[t])


.. parsed-literal::

    [[0.    0.167 0.107 0.534 0.313]
     [0.167 0.    0.195 0.821 0.204]
     [0.107 0.195 0.    0.005 0.105]
     [0.534 0.821 0.005 0.    0.049]
     [0.313 0.204 0.105 0.049 0.   ]]
    

.. code-block:: python

    t = 2
    print(p_values_list[2])


.. parsed-literal::

    [[0.    0.723 0.596 0.579 0.564]
     [0.723 0.    0.612 0.688 0.412]
     [0.596 0.612 0.    0.267 0.636]
     [0.579 0.688 0.267 0.    0.421]
     [0.564 0.412 0.636 0.421 0.   ]]
    

Bootstrapping
-------------

We call :func:`~lingam.LongitudinalLiNGAM.bootstrap` method instead of :func:`~lingam.LongitudinalLiNGAM.fit`. Here, the second argument specifies the number of bootstrap sampling.

.. code-block:: python

    model = lingam.LongitudinalLiNGAM()
    result = model.bootstrap(X_t, n_sampling=100)

Causal Directions
-----------------

Since :class:`~lingam.LongitudinalBootstrapResult` object is returned, we can get the ranking of the causal directions extracted by :func:`~lingam.LongitudinalBootstrapResult.get_causal_direction_counts` method. In the following sample code, ``n_directions`` option is limited to the causal directions of the top 8 rankings, and ``min_causal_effect`` option is limited to causal directions with a coefficient of 0.01 or more.

.. code-block:: python

    cdc_list = result.get_causal_direction_counts(n_directions=12, min_causal_effect=0.01, split_by_causal_effect_sign=True)

.. code-block:: python

    t = 1
    labels = [f'x{i}({u})' for u in [t, t-1] for i in range(5)]
    print_causal_directions(cdc_list[t], 100, labels=labels)


.. parsed-literal::

    x4(1) <--- x4(0) (b>0) (100.0%)
    x2(1) <--- x0(0) (b<0) (100.0%)
    x3(1) <--- x0(0) (b>0) (100.0%)
    x1(1) <--- x3(0) (b>0) (100.0%)
    x1(1) <--- x2(0) (b<0) (100.0%)
    x3(1) <--- x2(0) (b<0) (100.0%)
    x3(1) <--- x4(0) (b<0) (100.0%)
    x1(1) <--- x3(1) (b>0) (100.0%)
    x0(1) <--- x4(1) (b<0) (100.0%)
    x4(1) <--- x1(0) (b<0) (100.0%)
    x4(1) <--- x1(1) (b<0) (100.0%)
    x2(1) <--- x2(0) (b>0) (100.0%)
    

.. code-block:: python

    t = 2
    labels = [f'x{i}({u})' for u in [t, t-1] for i in range(5)]
    print_causal_directions(cdc_list[t], 100, labels=labels)


.. parsed-literal::

    x0(2) <--- x0(1) (b>0) (100.0%)
    x4(2) <--- x1(1) (b>0) (100.0%)
    x3(2) <--- x2(1) (b>0) (100.0%)
    x3(2) <--- x1(1) (b<0) (100.0%)
    x3(2) <--- x0(1) (b>0) (100.0%)
    x3(2) <--- x2(2) (b<0) (100.0%)
    x2(2) <--- x3(1) (b>0) (100.0%)
    x2(2) <--- x1(1) (b>0) (100.0%)
    x4(2) <--- x3(1) (b>0) (100.0%)
    x1(2) <--- x3(1) (b>0) (100.0%)
    x1(2) <--- x2(1) (b>0) (100.0%)
    x1(2) <--- x0(1) (b<0) (100.0%)
    

Directed Acyclic Graphs
-----------------------

Also, using the :func:`~lingam.LongitudinalBootstrapResult.get_directed_acyclic_graph_counts` method, we can get the ranking of the DAGs extracted. In the following sample code, ``n_dags`` option is limited to the dags of the top 3 rankings, and ``min_causal_effect`` option is limited to causal directions with a coefficient of 0.01 or more.

.. code-block:: python

    dagc_list = result.get_directed_acyclic_graph_counts(n_dags=3, min_causal_effect=0.01, split_by_causal_effect_sign=True)

.. code-block:: python

    t = 1
    labels = [f'x{i}({u})' for u in [t, t-1] for i in range(5)]
    print_dagc(dagc_list[t], 100, labels=labels)


.. parsed-literal::

    DAG[0]: 2.0%
    	x0(1) <--- x4(1) (b<0)
    	x0(1) <--- x0(0) (b>0)
    	x0(1) <--- x1(0) (b>0)
    	x0(1) <--- x2(0) (b<0)
    	x0(1) <--- x3(0) (b>0)
    	x0(1) <--- x4(0) (b<0)
    	x1(1) <--- x3(1) (b>0)
    	x1(1) <--- x0(0) (b>0)
    	x1(1) <--- x1(0) (b<0)
    	x1(1) <--- x2(0) (b<0)
    	x1(1) <--- x3(0) (b>0)
    	x1(1) <--- x4(0) (b>0)
    	x2(1) <--- x1(1) (b>0)
    	x2(1) <--- x0(0) (b<0)
    	x2(1) <--- x1(0) (b>0)
    	x2(1) <--- x2(0) (b>0)
    	x2(1) <--- x3(0) (b>0)
    	x2(1) <--- x4(0) (b>0)
    	x3(1) <--- x0(0) (b>0)
    	x3(1) <--- x1(0) (b<0)
    	x3(1) <--- x2(0) (b<0)
    	x3(1) <--- x4(0) (b<0)
    	x4(1) <--- x1(1) (b<0)
    	x4(1) <--- x0(0) (b<0)
    	x4(1) <--- x1(0) (b<0)
    	x4(1) <--- x2(0) (b<0)
    	x4(1) <--- x3(0) (b<0)
    	x4(1) <--- x4(0) (b>0)
    DAG[1]: 1.0%
    	x0(1) <--- x2(1) (b<0)
    	x0(1) <--- x4(1) (b<0)
    	x0(1) <--- x0(0) (b>0)
    	x0(1) <--- x1(0) (b<0)
    	x0(1) <--- x2(0) (b<0)
    	x0(1) <--- x3(0) (b>0)
    	x0(1) <--- x4(0) (b<0)
    	x1(1) <--- x3(1) (b>0)
    	x1(1) <--- x0(0) (b>0)
    	x1(1) <--- x1(0) (b<0)
    	x1(1) <--- x2(0) (b<0)
    	x1(1) <--- x3(0) (b>0)
    	x1(1) <--- x4(0) (b>0)
    	x2(1) <--- x1(1) (b>0)
    	x2(1) <--- x0(0) (b<0)
    	x2(1) <--- x2(0) (b>0)
    	x2(1) <--- x3(0) (b>0)
    	x2(1) <--- x4(0) (b>0)
    	x3(1) <--- x0(0) (b>0)
    	x3(1) <--- x1(0) (b>0)
    	x3(1) <--- x2(0) (b<0)
    	x3(1) <--- x3(0) (b<0)
    	x3(1) <--- x4(0) (b<0)
    	x4(1) <--- x1(1) (b<0)
    	x4(1) <--- x2(1) (b<0)
    	x4(1) <--- x3(1) (b>0)
    	x4(1) <--- x0(0) (b<0)
    	x4(1) <--- x1(0) (b<0)
    	x4(1) <--- x2(0) (b>0)
    	x4(1) <--- x3(0) (b>0)
    	x4(1) <--- x4(0) (b>0)
    DAG[2]: 1.0%
    	x0(1) <--- x1(1) (b>0)
    	x0(1) <--- x4(1) (b<0)
    	x0(1) <--- x1(0) (b>0)
    	x0(1) <--- x2(0) (b<0)
    	x0(1) <--- x3(0) (b>0)
    	x0(1) <--- x4(0) (b<0)
    	x1(1) <--- x3(1) (b>0)
    	x1(1) <--- x0(0) (b>0)
    	x1(1) <--- x1(0) (b<0)
    	x1(1) <--- x2(0) (b<0)
    	x1(1) <--- x3(0) (b>0)
    	x1(1) <--- x4(0) (b>0)
    	x2(1) <--- x1(1) (b>0)
    	x2(1) <--- x0(0) (b<0)
    	x2(1) <--- x1(0) (b>0)
    	x2(1) <--- x2(0) (b>0)
    	x2(1) <--- x3(0) (b>0)
    	x2(1) <--- x4(0) (b>0)
    	x3(1) <--- x0(0) (b>0)
    	x3(1) <--- x1(0) (b<0)
    	x3(1) <--- x2(0) (b<0)
    	x3(1) <--- x3(0) (b<0)
    	x3(1) <--- x4(0) (b<0)
    	x4(1) <--- x1(1) (b<0)
    	x4(1) <--- x2(1) (b<0)
    	x4(1) <--- x3(1) (b>0)
    	x4(1) <--- x0(0) (b<0)
    	x4(1) <--- x1(0) (b<0)
    	x4(1) <--- x2(0) (b<0)
    	x4(1) <--- x3(0) (b<0)
    	x4(1) <--- x4(0) (b>0)
    

.. code-block:: python

    t = 2
    labels = [f'x{i}({u})' for u in [t, t-1] for i in range(5)]
    print_dagc(dagc_list[t], 100, labels=labels)


.. parsed-literal::

    DAG[0]: 3.0%
    	x0(2) <--- x0(1) (b>0)
    	x0(2) <--- x1(1) (b>0)
    	x0(2) <--- x2(1) (b>0)
    	x0(2) <--- x3(1) (b<0)
    	x0(2) <--- x4(1) (b>0)
    	x1(2) <--- x2(2) (b<0)
    	x1(2) <--- x4(2) (b>0)
    	x1(2) <--- x0(1) (b<0)
    	x1(2) <--- x1(1) (b<0)
    	x1(2) <--- x2(1) (b>0)
    	x1(2) <--- x3(1) (b>0)
    	x1(2) <--- x4(1) (b>0)
    	x2(2) <--- x0(2) (b>0)
    	x2(2) <--- x0(1) (b>0)
    	x2(2) <--- x1(1) (b>0)
    	x2(2) <--- x2(1) (b<0)
    	x2(2) <--- x3(1) (b>0)
    	x2(2) <--- x4(1) (b<0)
    	x3(2) <--- x2(2) (b<0)
    	x3(2) <--- x0(1) (b>0)
    	x3(2) <--- x1(1) (b<0)
    	x3(2) <--- x2(1) (b>0)
    	x3(2) <--- x3(1) (b>0)
    	x3(2) <--- x4(1) (b<0)
    	x4(2) <--- x0(2) (b>0)
    	x4(2) <--- x0(1) (b<0)
    	x4(2) <--- x1(1) (b>0)
    	x4(2) <--- x2(1) (b<0)
    	x4(2) <--- x3(1) (b>0)
    	x4(2) <--- x4(1) (b<0)
    DAG[1]: 2.0%
    	x0(2) <--- x0(1) (b>0)
    	x0(2) <--- x1(1) (b>0)
    	x0(2) <--- x2(1) (b>0)
    	x0(2) <--- x3(1) (b<0)
    	x0(2) <--- x4(1) (b>0)
    	x1(2) <--- x2(2) (b<0)
    	x1(2) <--- x4(2) (b>0)
    	x1(2) <--- x0(1) (b<0)
    	x1(2) <--- x1(1) (b<0)
    	x1(2) <--- x2(1) (b>0)
    	x1(2) <--- x3(1) (b>0)
    	x1(2) <--- x4(1) (b<0)
    	x2(2) <--- x0(2) (b>0)
    	x2(2) <--- x0(1) (b>0)
    	x2(2) <--- x1(1) (b>0)
    	x2(2) <--- x2(1) (b<0)
    	x2(2) <--- x3(1) (b>0)
    	x2(2) <--- x4(1) (b>0)
    	x3(2) <--- x2(2) (b<0)
    	x3(2) <--- x0(1) (b>0)
    	x3(2) <--- x1(1) (b<0)
    	x3(2) <--- x2(1) (b>0)
    	x3(2) <--- x3(1) (b<0)
    	x3(2) <--- x4(1) (b<0)
    	x4(2) <--- x0(2) (b>0)
    	x4(2) <--- x0(1) (b<0)
    	x4(2) <--- x1(1) (b>0)
    	x4(2) <--- x2(1) (b<0)
    	x4(2) <--- x3(1) (b>0)
    	x4(2) <--- x4(1) (b<0)
    DAG[2]: 2.0%
    	x0(2) <--- x0(1) (b>0)
    	x0(2) <--- x1(1) (b>0)
    	x0(2) <--- x2(1) (b<0)
    	x0(2) <--- x3(1) (b<0)
    	x0(2) <--- x4(1) (b<0)
    	x1(2) <--- x2(2) (b<0)
    	x1(2) <--- x4(2) (b>0)
    	x1(2) <--- x0(1) (b<0)
    	x1(2) <--- x1(1) (b<0)
    	x1(2) <--- x2(1) (b>0)
    	x1(2) <--- x3(1) (b>0)
    	x1(2) <--- x4(1) (b>0)
    	x2(2) <--- x0(1) (b>0)
    	x2(2) <--- x1(1) (b>0)
    	x2(2) <--- x2(1) (b<0)
    	x2(2) <--- x3(1) (b>0)
    	x2(2) <--- x4(1) (b<0)
    	x3(2) <--- x2(2) (b<0)
    	x3(2) <--- x0(1) (b>0)
    	x3(2) <--- x1(1) (b<0)
    	x3(2) <--- x2(1) (b>0)
    	x3(2) <--- x3(1) (b<0)
    	x3(2) <--- x4(1) (b<0)
    	x4(2) <--- x0(2) (b>0)
    	x4(2) <--- x0(1) (b<0)
    	x4(2) <--- x1(1) (b>0)
    	x4(2) <--- x2(1) (b<0)
    	x4(2) <--- x3(1) (b>0)
    	x4(2) <--- x4(1) (b<0)
    

Probability
-----------

Using the :func:`~lingam.LongitudinalBootstrapResult.get_probabilities` method, we can get the probability of bootstrapping.

.. code-block:: python

    probs = result.get_probabilities(min_causal_effect=0.01)
    print(probs[1])


.. parsed-literal::

    [[[0.   0.51 0.09 0.15 1.  ]
      [0.   0.   0.   1.   0.  ]
      [0.02 0.99 0.   0.52 0.3 ]
      [0.   0.   0.   0.   0.  ]
      [0.   1.   0.23 0.3  0.  ]]
    
     [[0.92 0.97 1.   0.94 0.99]
      [0.99 0.99 1.   1.   0.94]
      [1.   0.97 1.   0.99 0.87]
      [1.   0.98 1.   0.92 1.  ]
      [1.   1.   1.   1.   1.  ]]]
    

.. code-block:: python

    t = 1
    print('B(1,1):')
    print(probs[t, 0])
    print('B(1,0):')
    print(probs[t, 1])
    
    t = 2
    print('B(2,2):')
    print(probs[t, 0])
    print('B(2,1):')
    print(probs[t, 1])


.. parsed-literal::

    B(1,1):
    [[0.   0.51 0.09 0.15 1.  ]
     [0.   0.   0.   1.   0.  ]
     [0.02 0.99 0.   0.52 0.3 ]
     [0.   0.   0.   0.   0.  ]
     [0.   1.   0.23 0.3  0.  ]]
    B(1,0):
    [[0.92 0.97 1.   0.94 0.99]
     [0.99 0.99 1.   1.   0.94]
     [1.   0.97 1.   0.99 0.87]
     [1.   0.98 1.   0.92 1.  ]
     [1.   1.   1.   1.   1.  ]]
    B(2,2):
    [[0.   0.   0.   0.   0.  ]
     [0.1  0.   1.   0.06 1.  ]
     [0.78 0.   0.   0.   0.13]
     [0.13 0.   1.   0.   0.16]
     [0.88 0.   0.   0.   0.  ]]
    B(2,1):
    [[1.   1.   0.91 1.   0.92]
     [1.   0.86 1.   1.   0.95]
     [0.95 1.   0.96 1.   0.8 ]
     [1.   1.   1.   0.92 1.  ]
     [0.99 1.   0.96 1.   1.  ]]
    

Total Causal Effects
--------------------

Using the ``get_total_causal_effects()`` method, we can get the list of
total causal effect. The total causal effects we can get are dictionary
type variable. We can display the list nicely by assigning it to
pandas.DataFrame. Also, we have replaced the variable index with a label
below.

.. code-block:: python

    causal_effects = result.get_total_causal_effects(min_causal_effect=0.01)
    
    df = pd.DataFrame(causal_effects)
    
    labels = [f'x{i}({t})' for t in range(3) for i in range(5)]
    df['from'] = df['from'].apply(lambda x : labels[x])
    df['to'] = df['to'].apply(lambda x : labels[x])
    df




.. raw:: html

    <div>
    <style scoped>
        .dataframe {
            font-family: verdana, arial, sans-serif;
            font-size: 11px;
            color: #333333;
            border-width: 1px;
            border-color: #B3B3B3;
            border-collapse: collapse;
        }
        .dataframe thead th {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #B3B3B3;
            background-color: #B3B3B3;
        }
        .dataframe tbody th {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #B3B3B3;
        }
        .dataframe tr:nth-child(even) th{
        background-color: #EAEAEA;
        }
        .dataframe tr:nth-child(even) td{
            background-color: #EAEAEA;
        }
        .dataframe td {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #B3B3B3;
            background-color: #ffffff;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>from</th>
          <th>to</th>
          <th>effect</th>
          <th>probability</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>x1(1)</td>
          <td>x0(1)</td>
          <td>0.269441</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>1</th>
          <td>x0(2)</td>
          <td>x4(2)</td>
          <td>0.119620</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>2</th>
          <td>x4(1)</td>
          <td>x4(2)</td>
          <td>-0.109855</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>3</th>
          <td>x3(1)</td>
          <td>x4(2)</td>
          <td>0.260481</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>4</th>
          <td>x1(1)</td>
          <td>x4(2)</td>
          <td>0.297682</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>5</th>
          <td>x2(2)</td>
          <td>x3(2)</td>
          <td>-0.394208</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>6</th>
          <td>x4(1)</td>
          <td>x3(2)</td>
          <td>-0.152984</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>7</th>
          <td>x3(1)</td>
          <td>x3(2)</td>
          <td>-0.284373</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>8</th>
          <td>x2(1)</td>
          <td>x3(2)</td>
          <td>0.425542</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>9</th>
          <td>x1(1)</td>
          <td>x3(2)</td>
          <td>-0.263069</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>10</th>
          <td>x0(2)</td>
          <td>x2(2)</td>
          <td>0.177046</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>11</th>
          <td>x4(1)</td>
          <td>x2(2)</td>
          <td>-0.110188</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>12</th>
          <td>x3(1)</td>
          <td>x2(2)</td>
          <td>0.524608</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>13</th>
          <td>x1(1)</td>
          <td>x2(2)</td>
          <td>0.329232</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>14</th>
          <td>x4(2)</td>
          <td>x1(2)</td>
          <td>0.113916</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>15</th>
          <td>x2(2)</td>
          <td>x1(2)</td>
          <td>-0.429614</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>16</th>
          <td>x0(1)</td>
          <td>x2(2)</td>
          <td>0.202225</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>17</th>
          <td>x1(1)</td>
          <td>x0(2)</td>
          <td>0.154852</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>18</th>
          <td>x1(1)</td>
          <td>x1(2)</td>
          <td>-0.145485</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>19</th>
          <td>x3(1)</td>
          <td>x0(1)</td>
          <td>0.116298</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>20</th>
          <td>x0(1)</td>
          <td>x1(2)</td>
          <td>-0.462228</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>21</th>
          <td>x4(1)</td>
          <td>x0(1)</td>
          <td>-0.562721</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>22</th>
          <td>x3(1)</td>
          <td>x0(2)</td>
          <td>-0.238794</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>23</th>
          <td>x3(1)</td>
          <td>x1(1)</td>
          <td>0.317693</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>24</th>
          <td>x4(1)</td>
          <td>x1(2)</td>
          <td>0.222208</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>25</th>
          <td>x1(1)</td>
          <td>x2(1)</td>
          <td>0.187445</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>26</th>
          <td>x1(1)</td>
          <td>x4(1)</td>
          <td>-0.280015</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>27</th>
          <td>x4(2)</td>
          <td>x3(2)</td>
          <td>-0.059277</td>
          <td>0.92</td>
        </tr>
        <tr>
          <th>28</th>
          <td>x4(1)</td>
          <td>x0(2)</td>
          <td>-0.139972</td>
          <td>0.91</td>
        </tr>
        <tr>
          <th>29</th>
          <td>x4(2)</td>
          <td>x2(2)</td>
          <td>0.033740</td>
          <td>0.69</td>
        </tr>
        <tr>
          <th>30</th>
          <td>x4(1)</td>
          <td>x2(1)</td>
          <td>-0.050954</td>
          <td>0.54</td>
        </tr>
        <tr>
          <th>31</th>
          <td>x2(1)</td>
          <td>x4(1)</td>
          <td>-0.102010</td>
          <td>0.46</td>
        </tr>
        <tr>
          <th>32</th>
          <td>x2(1)</td>
          <td>x0(2)</td>
          <td>0.034217</td>
          <td>0.35</td>
        </tr>
        <tr>
          <th>33</th>
          <td>x2(1)</td>
          <td>x1(2)</td>
          <td>0.161172</td>
          <td>0.34</td>
        </tr>
        <tr>
          <th>34</th>
          <td>x2(2)</td>
          <td>x4(2)</td>
          <td>0.029630</td>
          <td>0.31</td>
        </tr>
        <tr>
          <th>35</th>
          <td>x0(1)</td>
          <td>x3(2)</td>
          <td>0.106614</td>
          <td>0.19</td>
        </tr>
        <tr>
          <th>36</th>
          <td>x0(1)</td>
          <td>x0(2)</td>
          <td>0.136141</td>
          <td>0.15</td>
        </tr>
        <tr>
          <th>37</th>
          <td>x2(1)</td>
          <td>x2(2)</td>
          <td>-0.089162</td>
          <td>0.12</td>
        </tr>
        <tr>
          <th>38</th>
          <td>x3(2)</td>
          <td>x4(2)</td>
          <td>-0.081235</td>
          <td>0.08</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br>



We can easily perform sorting operations with pandas.DataFrame.

.. code-block:: python

    df.sort_values('effect', ascending=False).head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe {
            font-family: verdana, arial, sans-serif;
            font-size: 11px;
            color: #333333;
            border-width: 1px;
            border-color: #B3B3B3;
            border-collapse: collapse;
        }
        .dataframe thead th {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #B3B3B3;
            background-color: #B3B3B3;
        }
        .dataframe tbody th {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #B3B3B3;
        }
        .dataframe tr:nth-child(even) th{
        background-color: #EAEAEA;
        }
        .dataframe tr:nth-child(even) td{
            background-color: #EAEAEA;
        }
        .dataframe td {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #B3B3B3;
            background-color: #ffffff;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>from</th>
          <th>to</th>
          <th>effect</th>
          <th>probability</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>12</th>
          <td>x3(1)</td>
          <td>x2(2)</td>
          <td>0.524608</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>8</th>
          <td>x2(1)</td>
          <td>x3(2)</td>
          <td>0.425542</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>13</th>
          <td>x1(1)</td>
          <td>x2(2)</td>
          <td>0.329232</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>23</th>
          <td>x3(1)</td>
          <td>x1(1)</td>
          <td>0.317693</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>x1(1)</td>
          <td>x4(2)</td>
          <td>0.297682</td>
          <td>1.0</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br>



And with pandas.DataFrame, we can easily filter by keywords. The
following code extracts the causal direction towards x0(2).

.. code-block:: python

    df[df['to']=='x0(2)'].head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe {
            font-family: verdana, arial, sans-serif;
            font-size: 11px;
            color: #333333;
            border-width: 1px;
            border-color: #B3B3B3;
            border-collapse: collapse;
        }
        .dataframe thead th {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #B3B3B3;
            background-color: #B3B3B3;
        }
        .dataframe tbody th {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #B3B3B3;
        }
        .dataframe tr:nth-child(even) th{
        background-color: #EAEAEA;
        }
        .dataframe tr:nth-child(even) td{
            background-color: #EAEAEA;
        }
        .dataframe td {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #B3B3B3;
            background-color: #ffffff;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>from</th>
          <th>to</th>
          <th>effect</th>
          <th>probability</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>17</th>
          <td>x1(1)</td>
          <td>x0(2)</td>
          <td>0.154852</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>22</th>
          <td>x3(1)</td>
          <td>x0(2)</td>
          <td>-0.238794</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>28</th>
          <td>x4(1)</td>
          <td>x0(2)</td>
          <td>-0.139972</td>
          <td>0.91</td>
        </tr>
        <tr>
          <th>32</th>
          <td>x2(1)</td>
          <td>x0(2)</td>
          <td>0.034217</td>
          <td>0.35</td>
        </tr>
        <tr>
          <th>36</th>
          <td>x0(1)</td>
          <td>x0(2)</td>
          <td>0.136141</td>
          <td>0.15</td>
        </tr>
      </tbody>
    </table>
    </div>



Because it holds the raw data of the total causal effect (the original
data for calculating the median), it is possible to draw a histogram of
the values of the causal effect, as shown below.

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    %matplotlib inline
    
    from_index = 5 # index of x0(1). (index:0)+(n_features:5)*(timepoint:1) = 5
    to_index = 12 # index of x2(2). (index:2)+(n_features:5)*(timepoint:2) = 12
    plt.hist(result.total_effects_[:, to_index, from_index])


.. image:: ../image/longitudinal_hist.png

