
VARMALiNGAM
===========

Model
-------------------
VARMALiNGAM [3]_ is an extension of the basic LiNGAM model [1]_ to time series cases. 
It combines the basic LiNGAM model with the classic vector autoregressive moving average models (VARMA). 
It enables analyzing both lagged and contemporaneous (instantaneous) causal relations, whereas the classic VARMA only analyzes lagged causal relations. 
This VARMALiNGAM model also is an extension of the VARLiNGAM model [2]_. 
It uses VARMA to analyze lagged causal relations instead of VAR. 
This VARMALiNGAM makes the following assumptions similarly to the basic LiNGAM model [1]_:

#. Linearity
#. Non-Gaussian continuous error variables (except at most one)
#. Acyclicity of contemporaneous causal relations
#. No hidden common causes between contempraneous error variables

Denote observed variables at time point :math:`{t}` by :math:`{x}_{i}(t)` and error variables by :math:`{e}_{i}(t)`. 
Collect them in vectors :math:`{x}(t)` and :math:`{e}(t)`, respectivelly. 
Further, denote by matrices :math:`{B}_{\tau}` and :math:`{\Omega}_{\omega}` coefficient matrices with time lags :math:`{\tau}` and :math:`{\omega}`, respectivelly.


Due to the acyclicity assumption of contemporaneous causal relations, the adjacency matrix $B_0$ can be permuted to be strictly lower-triangular by a simultaneous row and column permutation.
The error variables :math:`{e}_{i}(t)` are independent due to the assumption of no hidden common causes. 

Then, mathematically, the model for observed variable vector :math:`{x}(t)` is written as 

$$ x(t) = \\sum_{ \\tau = 0}^k B_{ \\tau } x(t - \\tau) + e(t) - \\sum_{ \\omega = 1}^{\\ell} \\Omega_{ \\omega } e(t- \\omega).$$

Example applications are found `here <https://www.shimizulab.org/lingam/lingampapers/applications-and-tailor-made-methods>`__, especially in Section. Economics/Finance/Marketing. 
For example, [3]_ uses the VARLiNGAM model to to study the processes of firm growth and firm performance using microeconomic data 
and to analyse the effects of monetary policy using macroeconomic data. 

References

    .. [1] S. Shimizu, P. O. Hoyer, A. Hyvärinen, and A. J. Kerminen.
       A linear non-gaussian acyclic model for causal discovery.
       Journal of Machine Learning Research, 7:2003-2030, 2006.
    .. [2] A. Hyvärinen, K. Zhang, S. Shimizu, and P. O. Hoyer. 
        Estimation of a structural vector autoregression model using non-Gaussianity. 
        Journal of Machine Learning Research, 11: 1709-1731, 2010.
    .. [3] Y. Kawahara, S. Shimizu and T. Washio. 
        Analyzing relationships among ARMA processes based on non-Gaussianity of external influences. 
        Neurocomputing, 74(12-13): 2212-2221, 2011. [PDF]


Import and settings
-------------------

In this example, we need to import ``numpy``, ``pandas``, and
``graphviz`` in addition to ``lingam``.

.. code-block:: python

    import numpy as np
    import pandas as pd
    import graphviz
    import lingam
    from lingam.utils import make_dot, print_causal_directions, print_dagc
    
    import warnings
    warnings.filterwarnings('ignore')
    
    print([np.__version__, pd.__version__, graphviz.__version__, lingam.__version__])
    
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(0)


.. parsed-literal::

    ['1.24.4', '2.0.3', '0.20.1', '1.8.3']


Test data
---------

We create test data consisting of 5 variables.

.. code-block:: python

    psi0 = np.array([
        [ 0.  ,  0.  , -0.25,  0.  ,  0.  ],
        [-0.38,  0.  ,  0.14,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.44, -0.2 , -0.09,  0.  ,  0.  ],
        [ 0.07, -0.06,  0.  ,  0.07,  0.  ]
    ])
    phi1 = np.array([
        [-0.04, -0.29, -0.26,  0.14,  0.47],
        [-0.42,  0.2 ,  0.1 ,  0.24,  0.25],
        [-0.25,  0.18, -0.06,  0.15,  0.18],
        [ 0.22,  0.39,  0.08,  0.12, -0.37],
        [-0.43,  0.09, -0.23,  0.16,  0.25]
    ])
    theta1 = np.array([
        [ 0.15, -0.02, -0.3 , -0.2 ,  0.21],
        [ 0.32,  0.12, -0.11,  0.03,  0.42],
        [-0.07, -0.5 ,  0.03, -0.27, -0.21],
        [-0.17,  0.35,  0.25,  0.24, -0.25],
        [ 0.09,  0.4 ,  0.41,  0.24, -0.31]
    ])
    causal_order = [2, 0, 1, 3, 4]
    
    # data generated from psi0 and phi1 and theta1, causal_order
    X = np.loadtxt('data/sample_data_varma_lingam.csv', delimiter=',')

Causal Discovery
----------------

To run causal discovery, we create a :class:`~lingam.VARMALiNGAM` object and call the :func:`~lingam.VARMALiNGAM.fit` method.

.. code-block:: python

    model = lingam.VARMALiNGAM(order=(1, 1), criterion=None)
    model.fit(X)




.. parsed-literal::

    <lingam.varma_lingam.VARMALiNGAM at 0x1acfc3fa6d8>



Using the :attr:`~lingam.VARMALiNGAM.causal_order_` properties, we can see the causal ordering as a result of the causal discovery.

.. code-block:: python

    model.causal_order_




.. parsed-literal::

    [2, 0, 1, 3, 4]



Also, using the :attr:`~lingam.VARMALiNGAM.adjacency_matrices_` properties, we can see the adjacency matrix as a result of the causal discovery.

.. code-block:: python

    # psi0
    model.adjacency_matrices_[0][0]




.. parsed-literal::

    array([[ 0.   ,  0.   , -0.194,  0.   ,  0.   ],
           [-0.354,  0.   ,  0.191,  0.   ,  0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
           [ 0.558, -0.228,  0.   ,  0.   ,  0.   ],
           [ 0.115,  0.   ,  0.   ,  0.   ,  0.   ]])



.. code-block:: python

    # psi1
    model.adjacency_matrices_[0][1]




.. parsed-literal::

    array([[ 0.   , -0.394, -0.509,  0.   ,  0.659],
           [-0.3  ,  0.   ,  0.   ,  0.211,  0.404],
           [-0.281,  0.21 ,  0.   ,  0.118,  0.25 ],
           [ 0.082,  0.762,  0.178,  0.137, -0.819],
           [-0.507,  0.   , -0.278,  0.   ,  0.336]])



.. code-block:: python

    # omega0
    model.adjacency_matrices_[1][0]




.. parsed-literal::

    array([[ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
           [ 0.209,  0.365,  0.   ,  0.   ,  0.531],
           [ 0.   , -0.579, -0.105, -0.298, -0.235],
           [ 0.   ,  0.171,  0.414,  0.302,  0.   ],
           [ 0.297,  0.435,  0.482,  0.376, -0.438]])



Using ``DirectLiNGAM`` for the ``residuals_`` properties, we can
calculate psi0 matrix.

.. code-block:: python

    dlingam = lingam.DirectLiNGAM()
    dlingam.fit(model.residuals_)
    dlingam.adjacency_matrix_




.. parsed-literal::

    array([[ 0.   ,  0.   , -0.238,  0.   ,  0.   ],
           [-0.392,  0.   ,  0.182,  0.   ,  0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
           [ 0.587, -0.209,  0.   ,  0.   ,  0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ]])



We can draw a causal graph by utility funciton

.. code-block:: python

    labels = ['y0(t)', 'y1(t)', 'y2(t)', 'y3(t)', 'y4(t)', 'y0(t-1)', 'y1(t-1)', 'y2(t-1)', 'y3(t-1)', 'y4(t-1)']
    make_dot(np.hstack(model.adjacency_matrices_[0]), lower_limit=0.3, ignore_shape=True, labels=labels)




.. image:: ../image/varma_dag.svg


Independence between error variables
------------------------------------

To check if the LiNGAM assumption is broken, we can get p-values of
independence between error variables. The value in the i-th row and j-th
column of the obtained matrix shows the p-value of the independence of
the error variables :math:`e_i` and :math:`e_j`.

.. code-block:: python

    p_values = model.get_error_independence_p_values()
    print(p_values)


.. parsed-literal::

    [[0.    0.622 0.388 0.    0.539]
     [0.622 0.    0.087 0.469 0.069]
     [0.388 0.087 0.    0.248 0.229]
     [0.    0.469 0.248 0.    0.021]
     [0.539 0.069 0.229 0.021 0.   ]]


Bootstrap
---------

Bootstrapping
~~~~~~~~~~~~~

We call :func:`~lingam.VARMALiNGAM.bootstrap` method instead of :func:`~lingam.VARMALiNGAM.fit`. Here, the second argument specifies the number of bootstrap sampling.

.. code-block:: python

    model = lingam.VARMALiNGAM()
    result = model.bootstrap(X, n_sampling=100)

Causal Directions
-----------------

Since :class:`~lingam.BootstrapResult` object is returned, we can get the ranking of the causal directions extracted by :func:`~lingam.BootstrapResult.get_causal_direction_counts` method. In the following sample code, ``n_directions`` option is limited to the causal directions of the top 8 rankings, and ``min_causal_effect`` option is limited to causal directions with a coefficient of 0.4 or more.

.. code-block:: python

    cdc = result.get_causal_direction_counts(n_directions=8, min_causal_effect=0.4, split_by_causal_effect_sign=True)

We can check the result by utility function.

.. code-block:: python

    labels = ['y0(t)', 'y1(t)', 'y2(t)', 'y3(t)', 'y4(t)', 'y0(t-1)', 'y1(t-1)', 'y2(t-1)', 'y3(t-1)', 'y4(t-1)', 'e0(t-1)', 'e1(t-1)', 'e2(t-1)', 'e3(t-1)', 'e4(t-1)']
    print_causal_directions(cdc, 100, labels=labels)


.. parsed-literal::

    y3(t) <--- y4(t-1) (b<0) (98.0%)
    y3(t) <--- y1(t-1) (b>0) (98.0%)
    y0(t) <--- y4(t-1) (b>0) (96.0%)
    y1(t) <--- e4(t-1) (b>0) (91.0%)
    y2(t) <--- e1(t-1) (b<0) (80.0%)
    y4(t) <--- e2(t-1) (b>0) (71.0%)
    y1(t) <--- e0(t-1) (b>0) (64.0%)
    y2(t) <--- e4(t-1) (b<0) (62.0%)


Directed Acyclic Graphs
-----------------------

Also, using the :func:`~lingam.BootstrapResult.get_directed_acyclic_graph_counts` method, we can get the ranking of the DAGs extracted. In the following sample code, ``n_dags`` option is limited to the dags of the top 3 rankings, and ``min_causal_effect`` option is limited to causal directions with a coefficient of 0.3 or more.

.. code-block:: python

    dagc = result.get_directed_acyclic_graph_counts(n_dags=3, min_causal_effect=0.3, split_by_causal_effect_sign=True)

We can check the result by utility function.

.. code-block:: python

    print_dagc(dagc, 100, labels=labels)


.. parsed-literal::

    DAG[0]: 1.0%
    	y0(t) <--- y1(t) (b<0)
    	y0(t) <--- y1(t-1) (b<0)
    	y0(t) <--- y2(t-1) (b<0)
    	y0(t) <--- y4(t-1) (b>0)
    	y0(t) <--- e0(t-1) (b>0)
    	y0(t) <--- e1(t-1) (b<0)
    	y0(t) <--- e3(t-1) (b>0)
    	y0(t) <--- e4(t-1) (b>0)
    	y1(t) <--- y2(t) (b>0)
    	y1(t) <--- y2(t-1) (b>0)
    	y1(t) <--- y3(t-1) (b>0)
    	y1(t) <--- y4(t-1) (b<0)
    	y1(t) <--- e0(t-1) (b>0)
    	y1(t) <--- e2(t-1) (b>0)
    	y1(t) <--- e3(t-1) (b>0)
    	y1(t) <--- e4(t-1) (b>0)
    	y2(t) <--- y4(t-1) (b>0)
    	y2(t) <--- e0(t-1) (b<0)
    	y2(t) <--- e2(t-1) (b<0)
    	y2(t) <--- e3(t-1) (b<0)
    	y2(t) <--- e4(t-1) (b<0)
    	y3(t) <--- y1(t) (b<0)
    	y3(t) <--- y2(t) (b>0)
    	y3(t) <--- y1(t-1) (b>0)
    	y3(t) <--- y2(t-1) (b>0)
    	y3(t) <--- y3(t-1) (b>0)
    	y3(t) <--- y4(t-1) (b<0)
    	y3(t) <--- e0(t-1) (b>0)
    	y3(t) <--- e2(t-1) (b>0)
    	y3(t) <--- e3(t-1) (b>0)
    	y3(t) <--- e4(t-1) (b>0)
    	y4(t) <--- y0(t) (b>0)
    	y4(t) <--- y1(t) (b>0)
    	y4(t) <--- y3(t) (b>0)
    	y4(t) <--- y1(t-1) (b>0)
    	y4(t) <--- y2(t-1) (b>0)
    	y4(t) <--- y3(t-1) (b>0)
    	y4(t) <--- y4(t-1) (b<0)
    	y4(t) <--- e0(t-1) (b<0)
    	y4(t) <--- e1(t-1) (b>0)
    	y4(t) <--- e2(t-1) (b>0)
    	y4(t) <--- e3(t-1) (b<0)
    	y4(t) <--- e4(t-1) (b<0)
    DAG[1]: 1.0%
    	y0(t) <--- y1(t-1) (b<0)
    	y0(t) <--- y2(t-1) (b<0)
    	y0(t) <--- y4(t-1) (b>0)
    	y0(t) <--- e0(t-1) (b>0)
    	y1(t) <--- y3(t) (b<0)
    	y1(t) <--- y0(t-1) (b<0)
    	y1(t) <--- y1(t-1) (b>0)
    	y1(t) <--- e0(t-1) (b>0)
    	y1(t) <--- e1(t-1) (b>0)
    	y1(t) <--- e4(t-1) (b>0)
    	y2(t) <--- y1(t) (b>0)
    	y2(t) <--- y3(t) (b>0)
    	y2(t) <--- y4(t-1) (b>0)
    	y2(t) <--- e0(t-1) (b<0)
    	y2(t) <--- e1(t-1) (b<0)
    	y2(t) <--- e3(t-1) (b>0)
    	y2(t) <--- e4(t-1) (b<0)
    	y3(t) <--- y0(t) (b>0)
    	y3(t) <--- y1(t-1) (b>0)
    	y3(t) <--- y4(t-1) (b<0)
    	y3(t) <--- e0(t-1) (b<0)
    	y3(t) <--- e1(t-1) (b>0)
    	y3(t) <--- e2(t-1) (b>0)
    	y4(t) <--- y0(t) (b>0)
    	y4(t) <--- y0(t-1) (b<0)
    	y4(t) <--- y1(t-1) (b>0)
    	y4(t) <--- y4(t-1) (b<0)
    	y4(t) <--- e0(t-1) (b<0)
    	y4(t) <--- e1(t-1) (b>0)
    	y4(t) <--- e2(t-1) (b>0)
    DAG[2]: 1.0%
    	y0(t) <--- y1(t-1) (b<0)
    	y0(t) <--- y2(t-1) (b<0)
    	y0(t) <--- y4(t-1) (b>0)
    	y0(t) <--- e0(t-1) (b>0)
    	y1(t) <--- y0(t) (b<0)
    	y1(t) <--- y2(t) (b>0)
    	y1(t) <--- y4(t-1) (b>0)
    	y1(t) <--- e0(t-1) (b>0)
    	y1(t) <--- e1(t-1) (b>0)
    	y1(t) <--- e2(t-1) (b>0)
    	y1(t) <--- e3(t-1) (b>0)
    	y1(t) <--- e4(t-1) (b>0)
    	y2(t) <--- y0(t) (b<0)
    	y2(t) <--- y0(t-1) (b<0)
    	y2(t) <--- y4(t-1) (b>0)
    	y2(t) <--- e1(t-1) (b<0)
    	y2(t) <--- e2(t-1) (b<0)
    	y2(t) <--- e3(t-1) (b<0)
    	y2(t) <--- e4(t-1) (b<0)
    	y3(t) <--- y1(t) (b<0)
    	y3(t) <--- y2(t) (b>0)
    	y3(t) <--- y1(t-1) (b>0)
    	y3(t) <--- y2(t-1) (b>0)
    	y3(t) <--- y3(t-1) (b>0)
    	y3(t) <--- y4(t-1) (b<0)
    	y3(t) <--- e1(t-1) (b>0)
    	y3(t) <--- e2(t-1) (b>0)
    	y3(t) <--- e3(t-1) (b>0)
    	y3(t) <--- e4(t-1) (b>0)
    	y4(t) <--- y0(t) (b>0)
    	y4(t) <--- y1(t) (b>0)
    	y4(t) <--- y3(t) (b>0)
    	y4(t) <--- y1(t-1) (b>0)
    	y4(t) <--- y2(t-1) (b>0)
    	y4(t) <--- y4(t-1) (b<0)
    	y4(t) <--- e0(t-1) (b<0)
    	y4(t) <--- e2(t-1) (b>0)
    	y4(t) <--- e4(t-1) (b<0)


Probability
-----------

Using the :func:`~lingam.BootstrapResult.get_probabilities` method, we can get the probability of bootstrapping.

.. code-block:: python

    prob = result.get_probabilities(min_causal_effect=0.1)
    print('Probability of psi0:\n', prob[0])
    print('Probability of psi1:\n', prob[1])
    print('Probability of omega1:\n', prob[2])


.. parsed-literal::

    Probability of psi0:
     [[0.   0.26 0.46 0.26 0.3 ]
     [0.53 0.   0.54 0.37 0.33]
     [0.22 0.41 0.   0.38 0.12]
     [0.6  0.62 0.35 0.   0.38]
     [0.69 0.28 0.27 0.43 0.  ]]
    Probability of psi1:
     [[0.41 1.   1.   0.2  1.  ]
     [0.64 0.63 0.6  0.83 0.85]
     [0.76 0.69 0.64 0.47 0.95]
     [0.55 1.   0.71 0.73 1.  ]
     [0.94 0.79 0.71 0.53 0.74]]
    Probability of omega1:
     [[0.76 0.35 0.54 0.43 0.47]
     [0.87 0.77 0.63 0.5  1.  ]
     [0.63 0.95 0.66 0.84 0.93]
     [0.41 0.85 0.88 0.68 0.49]
     [0.66 0.81 0.92 0.58 0.69]]


Total Causal Effects
--------------------

Using the ``get_total causal_effects()`` method, we can get the list of
total causal effect. The total causal effects we can get are dictionary
type variable. We can display the list nicely by assigning it to
pandas.DataFrame. Also, we have replaced the variable index with a label
below.

.. code-block:: python

    causal_effects = result.get_total_causal_effects(min_causal_effect=0.01)
    df = pd.DataFrame(causal_effects)
    
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
          <td>y0(t-1)</td>
          <td>y4(t)</td>
          <td>-0.454092</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>1</th>
          <td>y4(t-1)</td>
          <td>y3(t)</td>
          <td>-0.593869</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>2</th>
          <td>y1(t-1)</td>
          <td>y3(t)</td>
          <td>0.514145</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>3</th>
          <td>y1(t-1)</td>
          <td>y0(t)</td>
          <td>-0.357521</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>4</th>
          <td>y2(t-1)</td>
          <td>y0(t)</td>
          <td>-0.443562</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>5</th>
          <td>y4(t-1)</td>
          <td>y0(t)</td>
          <td>0.573678</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>6</th>
          <td>y4(t-1)</td>
          <td>y2(t)</td>
          <td>0.360151</td>
          <td>0.97</td>
        </tr>
        <tr>
          <th>7</th>
          <td>y4(t-1)</td>
          <td>y4(t)</td>
          <td>0.213879</td>
          <td>0.96</td>
        </tr>
        <tr>
          <th>8</th>
          <td>y1(t-1)</td>
          <td>y1(t)</td>
          <td>0.206331</td>
          <td>0.96</td>
        </tr>
        <tr>
          <th>9</th>
          <td>y4(t-1)</td>
          <td>y1(t)</td>
          <td>0.235616</td>
          <td>0.95</td>
        </tr>
        <tr>
          <th>10</th>
          <td>y0(t-1)</td>
          <td>y1(t)</td>
          <td>-0.364361</td>
          <td>0.95</td>
        </tr>
        <tr>
          <th>11</th>
          <td>y3(t-1)</td>
          <td>y1(t)</td>
          <td>0.250602</td>
          <td>0.94</td>
        </tr>
        <tr>
          <th>12</th>
          <td>y0(t-1)</td>
          <td>y2(t)</td>
          <td>-0.267122</td>
          <td>0.94</td>
        </tr>
        <tr>
          <th>13</th>
          <td>y2(t-1)</td>
          <td>y1(t)</td>
          <td>0.221743</td>
          <td>0.93</td>
        </tr>
        <tr>
          <th>14</th>
          <td>y2(t-1)</td>
          <td>y4(t)</td>
          <td>-0.213938</td>
          <td>0.92</td>
        </tr>
        <tr>
          <th>15</th>
          <td>y1(t-1)</td>
          <td>y4(t)</td>
          <td>0.095743</td>
          <td>0.92</td>
        </tr>
        <tr>
          <th>16</th>
          <td>y0(t-1)</td>
          <td>y3(t)</td>
          <td>0.177089</td>
          <td>0.91</td>
        </tr>
        <tr>
          <th>17</th>
          <td>y1(t-1)</td>
          <td>y2(t)</td>
          <td>0.135946</td>
          <td>0.90</td>
        </tr>
        <tr>
          <th>18</th>
          <td>y3(t-1)</td>
          <td>y3(t)</td>
          <td>0.150796</td>
          <td>0.88</td>
        </tr>
        <tr>
          <th>19</th>
          <td>y2(t-1)</td>
          <td>y3(t)</td>
          <td>-0.021971</td>
          <td>0.84</td>
        </tr>
        <tr>
          <th>20</th>
          <td>y3(t-1)</td>
          <td>y4(t)</td>
          <td>0.170749</td>
          <td>0.79</td>
        </tr>
        <tr>
          <th>21</th>
          <td>y2(t-1)</td>
          <td>y2(t)</td>
          <td>-0.137767</td>
          <td>0.77</td>
        </tr>
        <tr>
          <th>22</th>
          <td>y0(t-1)</td>
          <td>y0(t)</td>
          <td>-0.094192</td>
          <td>0.75</td>
        </tr>
        <tr>
          <th>23</th>
          <td>y0(t)</td>
          <td>y4(t)</td>
          <td>0.934934</td>
          <td>0.70</td>
        </tr>
        <tr>
          <th>24</th>
          <td>y3(t-1)</td>
          <td>y2(t)</td>
          <td>0.141032</td>
          <td>0.66</td>
        </tr>
        <tr>
          <th>25</th>
          <td>y0(t)</td>
          <td>y3(t)</td>
          <td>0.636926</td>
          <td>0.63</td>
        </tr>
        <tr>
          <th>26</th>
          <td>y1(t)</td>
          <td>y3(t)</td>
          <td>-0.296396</td>
          <td>0.63</td>
        </tr>
        <tr>
          <th>27</th>
          <td>y3(t-1)</td>
          <td>y0(t)</td>
          <td>-0.027274</td>
          <td>0.63</td>
        </tr>
        <tr>
          <th>28</th>
          <td>y0(t)</td>
          <td>y1(t)</td>
          <td>-0.469409</td>
          <td>0.61</td>
        </tr>
        <tr>
          <th>29</th>
          <td>y2(t)</td>
          <td>y1(t)</td>
          <td>0.815024</td>
          <td>0.59</td>
        </tr>
        <tr>
          <th>30</th>
          <td>y2(t)</td>
          <td>y3(t)</td>
          <td>-0.102868</td>
          <td>0.57</td>
        </tr>
        <tr>
          <th>31</th>
          <td>y2(t)</td>
          <td>y0(t)</td>
          <td>-0.180943</td>
          <td>0.53</td>
        </tr>
        <tr>
          <th>32</th>
          <td>y2(t)</td>
          <td>y4(t)</td>
          <td>-0.054386</td>
          <td>0.49</td>
        </tr>
        <tr>
          <th>33</th>
          <td>y4(t)</td>
          <td>y3(t)</td>
          <td>0.132928</td>
          <td>0.45</td>
        </tr>
        <tr>
          <th>34</th>
          <td>y3(t)</td>
          <td>y4(t)</td>
          <td>0.453095</td>
          <td>0.44</td>
        </tr>
        <tr>
          <th>35</th>
          <td>y0(t)</td>
          <td>y2(t)</td>
          <td>-0.149761</td>
          <td>0.42</td>
        </tr>
        <tr>
          <th>36</th>
          <td>y4(t)</td>
          <td>y1(t)</td>
          <td>0.119746</td>
          <td>0.41</td>
        </tr>
        <tr>
          <th>37</th>
          <td>y1(t)</td>
          <td>y2(t)</td>
          <td>0.564823</td>
          <td>0.41</td>
        </tr>
        <tr>
          <th>38</th>
          <td>y3(t)</td>
          <td>y1(t)</td>
          <td>-0.706491</td>
          <td>0.37</td>
        </tr>
        <tr>
          <th>39</th>
          <td>y1(t)</td>
          <td>y4(t)</td>
          <td>-0.038562</td>
          <td>0.37</td>
        </tr>
        <tr>
          <th>40</th>
          <td>y3(t)</td>
          <td>y2(t)</td>
          <td>0.111094</td>
          <td>0.35</td>
        </tr>
        <tr>
          <th>41</th>
          <td>y3(t)</td>
          <td>y0(t)</td>
          <td>0.311717</td>
          <td>0.34</td>
        </tr>
        <tr>
          <th>42</th>
          <td>y1(t)</td>
          <td>y0(t)</td>
          <td>-0.300326</td>
          <td>0.33</td>
        </tr>
        <tr>
          <th>43</th>
          <td>y4(t)</td>
          <td>y2(t)</td>
          <td>0.139237</td>
          <td>0.32</td>
        </tr>
        <tr>
          <th>44</th>
          <td>y4(t)</td>
          <td>y0(t)</td>
          <td>0.405747</td>
          <td>0.30</td>
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
          <th>23</th>
          <td>y0(t)</td>
          <td>y4(t)</td>
          <td>0.934934</td>
          <td>0.70</td>
        </tr>
        <tr>
          <th>29</th>
          <td>y2(t)</td>
          <td>y1(t)</td>
          <td>0.815024</td>
          <td>0.59</td>
        </tr>
        <tr>
          <th>25</th>
          <td>y0(t)</td>
          <td>y3(t)</td>
          <td>0.636926</td>
          <td>0.63</td>
        </tr>
        <tr>
          <th>5</th>
          <td>y4(t-1)</td>
          <td>y0(t)</td>
          <td>0.573678</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>37</th>
          <td>y1(t)</td>
          <td>y2(t)</td>
          <td>0.564823</td>
          <td>0.41</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br>



And with pandas.DataFrame, we can easily filter by keywords. The
following code extracts the causal direction towards y2(t).

.. code-block:: python

    df[df['to']=='y2(t)'].head()




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
          <th>6</th>
          <td>y4(t-1)</td>
          <td>y2(t)</td>
          <td>0.360151</td>
          <td>0.97</td>
        </tr>
        <tr>
          <th>12</th>
          <td>y0(t-1)</td>
          <td>y2(t)</td>
          <td>-0.267122</td>
          <td>0.94</td>
        </tr>
        <tr>
          <th>17</th>
          <td>y1(t-1)</td>
          <td>y2(t)</td>
          <td>0.135946</td>
          <td>0.90</td>
        </tr>
        <tr>
          <th>21</th>
          <td>y2(t-1)</td>
          <td>y2(t)</td>
          <td>-0.137767</td>
          <td>0.77</td>
        </tr>
        <tr>
          <th>24</th>
          <td>y3(t-1)</td>
          <td>y2(t)</td>
          <td>0.141032</td>
          <td>0.66</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br>



Because it holds the raw data of the causal effect (the original data
for calculating the median), it is possible to draw a histogram of the
values of the causal effect, as shown below.

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    %matplotlib inline
    
    from_index = 5 # index of y0(t-1). (index:0)+(n_features:5)*(lag:1) = 5
    to_index = 2 # index of y2(t). (index:2)+(n_features:5)*(lag:0) = 2
    plt.hist(result.total_effects_[:, to_index, from_index])


.. image:: ../image/varma_hist.png


