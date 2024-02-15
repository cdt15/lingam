
VARLiNGAM
=========

Model
-------------------
VARLiNGAM [2]_ is an extension of the basic LiNGAM model [1]_ to time series cases. 
It combines the basic LiNGAM model with the classic vector autoregressive models (VAR). 
It enables analyzing both lagged and contemporaneous (instantaneous) causal relations, whereas the classic VAR only analyzes lagged causal relations. 
This VARLiNGAM makes the following assumptions similarly to the basic LiNGAM model [1]_:

#. Linearity
#. Non-Gaussian continuous error variables (except at most one)
#. Acyclicity of contemporaneous causal relations
#. No hidden common causes

Denote observed variables at time point :math:`{t}` by :math:`{x}_{i}(t)` and error variables by :math:`{e}_{i}(t)`. 
Collect them in vectors :math:`{x}(t)` and :math:`{e}(t)`, respectivelly. 
Further, denote by matrices :math:`{B}_{\tau}` adjacency matrices with time lag :math:`{\tau}`.

Due to the acyclicity assumption of contemporaneous causal relations, the coefficient matrix :math:`{B}_{0}` can be permuted to be strictly lower-triangular by a simultaneous row and column permutation.
The error variables :math:`{e}_{i}(t)` are independent due to the assumption of no hidden common causes. 

Then, mathematically, the model for observed variable vector :math:`{x}(t)` is written as 

$$ x(t) = \\sum_{ \\tau = 0}^k B_{ \\tau } x(t - \\tau) + e(t).$$

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
    .. [3] A. Moneta, D. Entner, P. O. Hoyer and A. Coad. 
        Causal inference by independent component analysis: Theory and applications. 
        Oxford Bulletin of Economics and Statistics, 75(5): 705-730, 2013.

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
    
    print([np.__version__, pd.__version__, graphviz.__version__, lingam.__version__])
    
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(0)


.. parsed-literal::

    ['1.24.4', '2.0.3', '0.20.1', '1.8.3']


Test data
---------

We create test data consisting of 5 variables.

.. code-block:: python

    B0 = [
        [0,-0.12,0,0,0],
        [0,0,0,0,0],
        [-0.41,0.01,0,-0.02,0],
        [0.04,-0.22,0,0,0],
        [0.15,0,-0.03,0,0],
    ]
    B1 = [
        [-0.32,0,0.12,0.32,0],
        [0,-0.35,-0.1,-0.46,0.4],
        [0,0,0.37,0,0.46],
        [-0.38,-0.1,-0.24,0,-0.13],
        [0,0,0,0,0],
    ]
    causal_order = [1, 0, 3, 2, 4]
    
    # data generated from B0 and B1
    X = pd.read_csv('data/sample_data_var_lingam.csv')

Causal Discovery
----------------

To run causal discovery, we create a :class:`~lingam.VARLiNGAM` object and call the :func:`~lingam.VARLiNGAM.fit` method.

.. code-block:: python

    model = lingam.VARLiNGAM()
    model.fit(X)




.. parsed-literal::

    <lingam.var_lingam.VARLiNGAM at 0x7fc1a642d970>



Using the :attr:`~lingam.VARLiNGAM.causal_order_` properties, we can see the causal ordering as a result of the causal discovery.

.. code-block:: python

    model.causal_order_




.. parsed-literal::

    [1, 0, 3, 2, 4]



Also, using the :attr:`~lingam.VARLiNGAM.adjacency_matrices_` properties, we can see the adjacency matrix as a result of the causal discovery.

.. code-block:: python

    # B0
    model.adjacency_matrices_[0]




.. parsed-literal::

    array([[ 0.   , -0.136,  0.   ,  0.   ,  0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
           [-0.484,  0.   ,  0.   ,  0.   ,  0.   ],
           [ 0.075, -0.21 ,  0.   ,  0.   ,  0.   ],
           [ 0.168,  0.   ,  0.   ,  0.   ,  0.   ]])



.. code-block:: python

    # B1
    model.adjacency_matrices_[1]




.. parsed-literal::

    array([[-0.358,  0.   ,  0.073,  0.302,  0.   ],
           [ 0.   , -0.338, -0.154, -0.335,  0.423],
           [ 0.   ,  0.   ,  0.424,  0.112,  0.493],
           [-0.386, -0.1  , -0.266,  0.   , -0.159],
           [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ]])



.. code-block:: python

    model.residuals_




.. parsed-literal::

    array([[-0.308,  0.911, -1.152, -1.159,  0.179],
           [ 1.364,  1.713, -1.389, -0.265, -0.192],
           [-0.861,  0.249,  0.479, -1.557, -0.462],
           ...,
           [-1.202,  1.819,  0.99 , -0.855, -0.127],
           [-0.133,  1.23 , -0.445, -0.753,  1.096],
           [-0.069,  0.558,  0.21 , -0.863, -0.189]])



Using ``DirectLiNGAM`` for the ``residuals_`` properties, we can
calculate B0 matrix.

.. code-block:: python

    dlingam = lingam.DirectLiNGAM()
    dlingam.fit(model.residuals_)
    dlingam.adjacency_matrix_




.. parsed-literal::

    array([[ 0.   , -0.144,  0.   ,  0.   ,  0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
           [-0.456,  0.   ,  0.   ,  0.   ,  0.   ],
           [ 0.   , -0.22 ,  0.   ,  0.   ,  0.   ],
           [ 0.157,  0.   ,  0.   ,  0.   ,  0.   ]])



We can draw a causal graph by utility funciton.

.. code-block:: python

    labels = ['x0(t)', 'x1(t)', 'x2(t)', 'x3(t)', 'x4(t)', 'x0(t-1)', 'x1(t-1)', 'x2(t-1)', 'x3(t-1)', 'x4(t-1)']
    make_dot(np.hstack(model.adjacency_matrices_), ignore_shape=True, lower_limit=0.05, labels=labels)




.. image:: ../image/var_dag.svg



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

    [[0.    0.127 0.104 0.042 0.746]
     [0.127 0.    0.086 0.874 0.739]
     [0.104 0.086 0.    0.404 0.136]
     [0.042 0.874 0.404 0.    0.763]
     [0.746 0.739 0.136 0.763 0.   ]]


Bootstrap
---------

Bootstrapping
~~~~~~~~~~~~~

We call :func:`~lingam.VARLiNGAM.bootstrap` method instead of :func:`~lingam.VARLiNGAM.fit`. Here, the second argument specifies the number of bootstrap sampling.

.. code-block:: python

    model = lingam.VARLiNGAM()
    result = model.bootstrap(X, n_sampling=100)

Causal Directions
-----------------

Since :class:`~lingam.BootstrapResult` object is returned, we can get the ranking of the causal directions extracted by :func:`~lingam.BootstrapResult.get_causal_direction_counts` method. In the following sample code, ``n_directions`` option is limited to the causal directions of the top 8 rankings, and ``min_causal_effect`` option is limited to causal directions with a coefficient of 0.3 or more.

.. code-block:: python

    cdc = result.get_causal_direction_counts(n_directions=8, min_causal_effect=0.3, split_by_causal_effect_sign=True)

We can check the result by utility function.

.. code-block:: python

    print_causal_directions(cdc, 100, labels=labels)


.. parsed-literal::

    x2(t) <--- x4(t-1) (b>0) (100.0%)
    x2(t) <--- x2(t-1) (b>0) (100.0%)
    x0(t) <--- x0(t-1) (b<0) (95.0%)
    x1(t) <--- x1(t-1) (b<0) (86.0%)
    x1(t) <--- x4(t-1) (b>0) (85.0%)
    x3(t) <--- x0(t-1) (b<0) (78.0%)
    x2(t) <--- x4(t) (b<0) (60.0%)
    x0(t) <--- x3(t-1) (b>0) (48.0%)


Directed Acyclic Graphs
-----------------------

Also, using the :func:`~lingam.BootstrapResult.get_directed_acyclic_graph_counts` method, we can get the ranking of the DAGs extracted. In the following sample code, ``n_dags`` option is limited to the dags of the top 3 rankings, and ``min_causal_effect`` option is limited to causal directions with a coefficient of 0.2 or more.

.. code-block:: python

    dagc = result.get_directed_acyclic_graph_counts(n_dags=3, min_causal_effect=0.2, split_by_causal_effect_sign=True)

We can check the result by utility function.

.. code-block:: python

    print_dagc(dagc, 100, labels=labels)


.. parsed-literal::

    DAG[0]: 5.0%
    	x0(t) <--- x0(t-1) (b<0)
    	x0(t) <--- x3(t-1) (b>0)
    	x1(t) <--- x0(t) (b<0)
    	x1(t) <--- x0(t-1) (b<0)
    	x1(t) <--- x1(t-1) (b<0)
    	x1(t) <--- x4(t-1) (b>0)
    	x2(t) <--- x0(t) (b<0)
    	x2(t) <--- x4(t) (b<0)
    	x2(t) <--- x2(t-1) (b>0)
    	x2(t) <--- x4(t-1) (b>0)
    	x3(t) <--- x0(t) (b>0)
    	x3(t) <--- x0(t-1) (b<0)
    	x3(t) <--- x2(t-1) (b<0)
    	x3(t) <--- x4(t-1) (b<0)
    DAG[1]: 5.0%
    	x0(t) <--- x0(t-1) (b<0)
    	x0(t) <--- x3(t-1) (b>0)
    	x1(t) <--- x0(t) (b<0)
    	x1(t) <--- x2(t) (b>0)
    	x1(t) <--- x0(t-1) (b<0)
    	x1(t) <--- x1(t-1) (b<0)
    	x1(t) <--- x4(t-1) (b>0)
    	x2(t) <--- x0(t) (b<0)
    	x2(t) <--- x4(t) (b<0)
    	x2(t) <--- x2(t-1) (b>0)
    	x2(t) <--- x4(t-1) (b>0)
    	x3(t) <--- x0(t) (b>0)
    	x3(t) <--- x0(t-1) (b<0)
    	x3(t) <--- x2(t-1) (b<0)
    DAG[2]: 5.0%
    	x0(t) <--- x0(t-1) (b<0)
    	x0(t) <--- x3(t-1) (b>0)
    	x1(t) <--- x1(t-1) (b<0)
    	x1(t) <--- x3(t-1) (b<0)
    	x1(t) <--- x4(t-1) (b>0)
    	x2(t) <--- x1(t) (b>0)
    	x2(t) <--- x3(t) (b>0)
    	x2(t) <--- x0(t-1) (b>0)
    	x2(t) <--- x2(t-1) (b>0)
    	x2(t) <--- x4(t-1) (b>0)
    	x3(t) <--- x1(t) (b<0)
    	x3(t) <--- x0(t-1) (b<0)
    	x3(t) <--- x2(t-1) (b<0)


Probability
-----------

Using the :func:`~lingam.BootstrapResult.get_probabilities` method, we can get the probability of bootstrapping.

.. code-block:: python

    prob = result.get_probabilities(min_causal_effect=0.1)
    print('Probability of B0:\n', prob[0])
    print('Probability of B1:\n', prob[1])


.. parsed-literal::

    Probability of B0:
     [[0.   0.6  0.04 0.06 0.14]
     [0.39 0.   0.25 0.18 0.16]
     [0.65 0.68 0.   0.67 0.84]
     [0.51 0.6  0.07 0.   0.66]
     [0.35 0.28 0.01 0.09 0.  ]]
    Probability of B1:
     [[1.   0.   0.3  1.   0.02]
     [0.56 1.   0.94 0.67 1.  ]
     [0.8  0.02 1.   0.25 1.  ]
     [1.   0.24 1.   0.08 1.  ]
     [0.02 0.   0.03 0.07 0.  ]]


Total Causal Effects
--------------------

Using the ``get_causal_effects()`` method, we can get the list of total
causal effect. The total causal effects we can get are dictionary type
variable. We can display the list nicely by assigning it to
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
          <td>x0(t-1)</td>
          <td>x2(t)</td>
          <td>0.181032</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>1</th>
          <td>x2(t-1)</td>
          <td>x2(t)</td>
          <td>0.388777</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>2</th>
          <td>x4(t-1)</td>
          <td>x1(t)</td>
          <td>0.427308</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>3</th>
          <td>x1(t-1)</td>
          <td>x1(t)</td>
          <td>-0.338691</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>4</th>
          <td>x0(t-1)</td>
          <td>x3(t)</td>
          <td>-0.397439</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>5</th>
          <td>x3(t-1)</td>
          <td>x0(t)</td>
          <td>0.345461</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>6</th>
          <td>x4(t-1)</td>
          <td>x2(t)</td>
          <td>0.501859</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>7</th>
          <td>x4(t-1)</td>
          <td>x3(t)</td>
          <td>-0.253700</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>8</th>
          <td>x0(t-1)</td>
          <td>x0(t)</td>
          <td>-0.357296</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>9</th>
          <td>x2(t-1)</td>
          <td>x3(t)</td>
          <td>-0.222886</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>10</th>
          <td>x3(t-1)</td>
          <td>x3(t)</td>
          <td>0.101008</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>11</th>
          <td>x3(t-1)</td>
          <td>x1(t)</td>
          <td>-0.315462</td>
          <td>0.99</td>
        </tr>
        <tr>
          <th>12</th>
          <td>x2(t-1)</td>
          <td>x0(t)</td>
          <td>0.090369</td>
          <td>0.99</td>
        </tr>
        <tr>
          <th>13</th>
          <td>x2(t-1)</td>
          <td>x1(t)</td>
          <td>-0.172693</td>
          <td>0.98</td>
        </tr>
        <tr>
          <th>14</th>
          <td>x1(t-1)</td>
          <td>x2(t)</td>
          <td>-0.063602</td>
          <td>0.89</td>
        </tr>
        <tr>
          <th>15</th>
          <td>x4(t)</td>
          <td>x2(t)</td>
          <td>-0.449165</td>
          <td>0.89</td>
        </tr>
        <tr>
          <th>16</th>
          <td>x3(t-1)</td>
          <td>x2(t)</td>
          <td>-0.079600</td>
          <td>0.89</td>
        </tr>
        <tr>
          <th>17</th>
          <td>x0(t)</td>
          <td>x2(t)</td>
          <td>-0.280635</td>
          <td>0.83</td>
        </tr>
        <tr>
          <th>18</th>
          <td>x1(t-1)</td>
          <td>x0(t)</td>
          <td>0.057164</td>
          <td>0.82</td>
        </tr>
        <tr>
          <th>19</th>
          <td>x4(t-1)</td>
          <td>x0(t)</td>
          <td>-0.050805</td>
          <td>0.79</td>
        </tr>
        <tr>
          <th>20</th>
          <td>x4(t)</td>
          <td>x3(t)</td>
          <td>-0.151835</td>
          <td>0.76</td>
        </tr>
        <tr>
          <th>21</th>
          <td>x1(t)</td>
          <td>x2(t)</td>
          <td>0.211957</td>
          <td>0.75</td>
        </tr>
        <tr>
          <th>22</th>
          <td>x1(t-1)</td>
          <td>x3(t)</td>
          <td>-0.021313</td>
          <td>0.75</td>
        </tr>
        <tr>
          <th>23</th>
          <td>x3(t)</td>
          <td>x2(t)</td>
          <td>0.248101</td>
          <td>0.66</td>
        </tr>
        <tr>
          <th>24</th>
          <td>x0(t)</td>
          <td>x3(t)</td>
          <td>0.259859</td>
          <td>0.64</td>
        </tr>
        <tr>
          <th>25</th>
          <td>x3(t-1)</td>
          <td>x4(t)</td>
          <td>0.061849</td>
          <td>0.62</td>
        </tr>
        <tr>
          <th>26</th>
          <td>x1(t)</td>
          <td>x3(t)</td>
          <td>-0.218490</td>
          <td>0.62</td>
        </tr>
        <tr>
          <th>27</th>
          <td>x1(t)</td>
          <td>x0(t)</td>
          <td>-0.199704</td>
          <td>0.61</td>
        </tr>
        <tr>
          <th>28</th>
          <td>x1(t)</td>
          <td>x4(t)</td>
          <td>-0.104466</td>
          <td>0.56</td>
        </tr>
        <tr>
          <th>29</th>
          <td>x0(t-1)</td>
          <td>x1(t)</td>
          <td>-0.119971</td>
          <td>0.53</td>
        </tr>
        <tr>
          <th>30</th>
          <td>x2(t-1)</td>
          <td>x4(t)</td>
          <td>0.017608</td>
          <td>0.50</td>
        </tr>
        <tr>
          <th>31</th>
          <td>x4(t-1)</td>
          <td>x4(t)</td>
          <td>-0.041991</td>
          <td>0.47</td>
        </tr>
        <tr>
          <th>32</th>
          <td>x1(t-1)</td>
          <td>x4(t)</td>
          <td>0.029382</td>
          <td>0.42</td>
        </tr>
        <tr>
          <th>33</th>
          <td>x0(t-1)</td>
          <td>x4(t)</td>
          <td>-0.055934</td>
          <td>0.42</td>
        </tr>
        <tr>
          <th>34</th>
          <td>x4(t)</td>
          <td>x1(t)</td>
          <td>-0.063677</td>
          <td>0.42</td>
        </tr>
        <tr>
          <th>35</th>
          <td>x4(t)</td>
          <td>x0(t)</td>
          <td>-0.066867</td>
          <td>0.40</td>
        </tr>
        <tr>
          <th>36</th>
          <td>x0(t)</td>
          <td>x1(t)</td>
          <td>-0.719255</td>
          <td>0.39</td>
        </tr>
        <tr>
          <th>37</th>
          <td>x0(t)</td>
          <td>x4(t)</td>
          <td>0.174717</td>
          <td>0.37</td>
        </tr>
        <tr>
          <th>38</th>
          <td>x2(t)</td>
          <td>x1(t)</td>
          <td>0.212699</td>
          <td>0.25</td>
        </tr>
        <tr>
          <th>39</th>
          <td>x3(t)</td>
          <td>x1(t)</td>
          <td>-0.308596</td>
          <td>0.20</td>
        </tr>
        <tr>
          <th>40</th>
          <td>x2(t)</td>
          <td>x3(t)</td>
          <td>-0.084192</td>
          <td>0.18</td>
        </tr>
        <tr>
          <th>41</th>
          <td>x3(t)</td>
          <td>x0(t)</td>
          <td>0.154238</td>
          <td>0.11</td>
        </tr>
        <tr>
          <th>42</th>
          <td>x3(t)</td>
          <td>x4(t)</td>
          <td>-0.205918</td>
          <td>0.10</td>
        </tr>
        <tr>
          <th>43</th>
          <td>x2(t)</td>
          <td>x0(t)</td>
          <td>-0.217316</td>
          <td>0.06</td>
        </tr>
        <tr>
          <th>44</th>
          <td>x2(t)</td>
          <td>x4(t)</td>
          <td>-0.093614</td>
          <td>0.03</td>
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
          <th>6</th>
          <td>x4(t-1)</td>
          <td>x2(t)</td>
          <td>0.501859</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>2</th>
          <td>x4(t-1)</td>
          <td>x1(t)</td>
          <td>0.427308</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>1</th>
          <td>x2(t-1)</td>
          <td>x2(t)</td>
          <td>0.388777</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>5</th>
          <td>x3(t-1)</td>
          <td>x0(t)</td>
          <td>0.345461</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>24</th>
          <td>x0(t)</td>
          <td>x3(t)</td>
          <td>0.259859</td>
          <td>0.64</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br>



And with pandas.DataFrame, we can easily filter by keywords. The following code extracts the causal direction towards x1(t).

.. code-block:: python

    df[df['to']=='x1(t)'].head()




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
          <th>2</th>
          <td>x4(t-1)</td>
          <td>x1(t)</td>
          <td>0.427308</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>3</th>
          <td>x1(t-1)</td>
          <td>x1(t)</td>
          <td>-0.338691</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>11</th>
          <td>x3(t-1)</td>
          <td>x1(t)</td>
          <td>-0.315462</td>
          <td>0.99</td>
        </tr>
        <tr>
          <th>13</th>
          <td>x2(t-1)</td>
          <td>x1(t)</td>
          <td>-0.172693</td>
          <td>0.98</td>
        </tr>
        <tr>
          <th>29</th>
          <td>x0(t-1)</td>
          <td>x1(t)</td>
          <td>-0.119971</td>
          <td>0.53</td>
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
    
    from_index = 7 # index of x2(t-1). (index:2)+(n_features:5)*(lag:1) = 7
    to_index = 2 # index of x2(t). (index:2)+(n_features:5)*(lag:0) = 2
    plt.hist(result.total_effects_[:, to_index, from_index])


.. image:: ../image/var_hist.png


