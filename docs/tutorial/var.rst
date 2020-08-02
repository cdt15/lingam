
VARLiNGAM
=========

Import and settings
-------------------

In this example, we need to import ``numpy``, ``pandas``, and ``graphviz`` in addition to ``lingam``.

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

    ['1.16.2', '0.24.2', '0.11.1', '1.3.1']
    

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

    <lingam.var_lingam.VARLiNGAM at 0x1f549305c88>



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

    array([[ 0.   , -0.144,  0.   ,  0.   ,  0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
           [-0.372,  0.   ,  0.   ,  0.   ,  0.   ],
           [ 0.069, -0.21 ,  0.   ,  0.   ,  0.   ],
           [ 0.083,  0.   , -0.033,  0.   ,  0.   ]])



.. code-block:: python

    # B1
    model.adjacency_matrices_[1]




.. parsed-literal::

    array([[-0.366, -0.011,  0.074,  0.297,  0.025],
           [-0.083, -0.349, -0.168, -0.327,  0.43 ],
           [ 0.077, -0.043,  0.427,  0.046,  0.49 ],
           [-0.389, -0.097, -0.263,  0.014, -0.159],
           [-0.018,  0.01 ,  0.001,  0.071,  0.003]])



We can draw a causal graph by utility funciton.

.. code-block:: python

    labels = ['x0(t)', 'x1(t)', 'x2(t)', 'x3(t)', 'x4(t)', 'x0(t-1)', 'x1(t-1)', 'x2(t-1)', 'x3(t-1)', 'x4(t-1)']
    make_dot(np.hstack(model.adjacency_matrices_), ignore_shape=True, lower_limit=0.05, labels=labels)




.. image:: ../image/var_dag.svg



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

    x0(t) <--- x0(t-1) (b<0) (100.0%)
    x1(t) <--- x1(t-1) (b<0) (100.0%)
    x1(t) <--- x3(t-1) (b<0) (100.0%)
    x1(t) <--- x4(t-1) (b>0) (100.0%)
    x2(t) <--- x2(t-1) (b>0) (100.0%)
    x2(t) <--- x4(t-1) (b>0) (100.0%)
    x3(t) <--- x0(t-1) (b<0) (100.0%)
    x2(t) <--- x0(t) (b<0) (99.0%)
    

Directed Acyclic Graphs
-----------------------

Also, using the :func:`~lingam.BootstrapResult.get_directed_acyclic_graph_counts` method, we can get the ranking of the DAGs extracted. In the following sample code, ``n_dags`` option is limited to the dags of the top 3 rankings, and ``min_causal_effect`` option is limited to causal directions with a coefficient of 0.2 or more.

.. code-block:: python

    dagc = result.get_directed_acyclic_graph_counts(n_dags=3, min_causal_effect=0.2, split_by_causal_effect_sign=True)

We can check the result by utility function.

.. code-block:: python

    print_dagc(dagc, 100, labels=labels)


.. parsed-literal::

    DAG[0]: 57.0%
    	x0(t) <--- x0(t-1) (b<0)
    	x0(t) <--- x3(t-1) (b>0)
    	x1(t) <--- x1(t-1) (b<0)
    	x1(t) <--- x3(t-1) (b<0)
    	x1(t) <--- x4(t-1) (b>0)
    	x2(t) <--- x0(t) (b<0)
    	x2(t) <--- x2(t-1) (b>0)
    	x2(t) <--- x4(t-1) (b>0)
    	x3(t) <--- x1(t) (b<0)
    	x3(t) <--- x0(t-1) (b<0)
    	x3(t) <--- x2(t-1) (b<0)
    DAG[1]: 42.0%
    	x0(t) <--- x0(t-1) (b<0)
    	x0(t) <--- x3(t-1) (b>0)
    	x1(t) <--- x1(t-1) (b<0)
    	x1(t) <--- x3(t-1) (b<0)
    	x1(t) <--- x4(t-1) (b>0)
    	x2(t) <--- x0(t) (b<0)
    	x2(t) <--- x2(t-1) (b>0)
    	x2(t) <--- x4(t-1) (b>0)
    	x3(t) <--- x0(t-1) (b<0)
    	x3(t) <--- x2(t-1) (b<0)
    DAG[2]: 1.0%
    	x0(t) <--- x0(t-1) (b<0)
    	x0(t) <--- x3(t-1) (b>0)
    	x1(t) <--- x1(t-1) (b<0)
    	x1(t) <--- x3(t-1) (b<0)
    	x1(t) <--- x4(t-1) (b>0)
    	x2(t) <--- x0(t) (b<0)
    	x2(t) <--- x2(t-1) (b>0)
    	x2(t) <--- x4(t-1) (b>0)
    	x3(t) <--- x1(t) (b<0)
    	x3(t) <--- x0(t-1) (b<0)
    	x3(t) <--- x2(t-1) (b<0)
    	x4(t) <--- x0(t) (b>0)
    

Probability
-----------

Using the :func:`~lingam.BootstrapResult.get_probabilities` method, we can get the probability of bootstrapping.

.. code-block:: python

    prob = result.get_probabilities(min_causal_effect=0.1)
    print('Probability of B0:\n', prob[0])
    print('Probability of B1:\n', prob[1])


.. parsed-literal::

    Probability of B0:
     [[0.   0.98 0.   0.02 0.  ]
     [0.   0.   0.   0.   0.  ]
     [1.   0.   0.   0.   0.01]
     [0.1  1.   0.   0.   0.  ]
     [0.51 0.   0.02 0.08 0.  ]]
    Probability of B1:
     [[1.   0.   0.02 1.   0.  ]
     [0.   1.   1.   1.   1.  ]
     [0.03 0.   1.   0.05 1.  ]
     [1.   0.16 1.   0.   1.  ]
     [0.   0.   0.   0.25 0.  ]]
    

Causal Effects
--------------

Using the :func:`~lingam.BootstrapResult.get_causal_effects` method, we can get the list of causal effect. The causal effects we can get are dictionary type variable. We can display the list nicely by assigning it to pandas.DataFrame. Also, we have replaced the variable index with a label below.

.. code-block:: python

    causal_effects = result.get_causal_effects(min_causal_effect=0.01)
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
          <td>x1(t)</td>
          <td>x0(t)</td>
          <td>-0.142773</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>1</th>
          <td>x4(t-1)</td>
          <td>x3(t)</td>
          <td>-0.245236</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>2</th>
          <td>x3(t-1)</td>
          <td>x3(t)</td>
          <td>0.114877</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>3</th>
          <td>x2(t-1)</td>
          <td>x3(t)</td>
          <td>-0.203598</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>4</th>
          <td>x0(t-1)</td>
          <td>x3(t)</td>
          <td>-0.324941</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>5</th>
          <td>x1(t)</td>
          <td>x3(t)</td>
          <td>-0.218320</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>6</th>
          <td>x4(t-1)</td>
          <td>x2(t)</td>
          <td>0.496761</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>7</th>
          <td>x1(t)</td>
          <td>x2(t)</td>
          <td>0.099477</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>8</th>
          <td>x0(t)</td>
          <td>x2(t)</td>
          <td>-0.439085</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>9</th>
          <td>x4(t-1)</td>
          <td>x1(t)</td>
          <td>0.454093</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>10</th>
          <td>x3(t-1)</td>
          <td>x1(t)</td>
          <td>-0.353886</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>11</th>
          <td>x2(t-1)</td>
          <td>x2(t)</td>
          <td>0.354316</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>12</th>
          <td>x1(t-1)</td>
          <td>x1(t)</td>
          <td>-0.294882</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>13</th>
          <td>x3(t-1)</td>
          <td>x0(t)</td>
          <td>0.339193</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>14</th>
          <td>x2(t-1)</td>
          <td>x0(t)</td>
          <td>0.107363</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>15</th>
          <td>x2(t-1)</td>
          <td>x1(t)</td>
          <td>-0.192527</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>16</th>
          <td>x0(t-1)</td>
          <td>x0(t)</td>
          <td>-0.381328</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>17</th>
          <td>x3(t-1)</td>
          <td>x4(t)</td>
          <td>0.099357</td>
          <td>0.99</td>
        </tr>
        <tr>
          <th>18</th>
          <td>x0(t)</td>
          <td>x4(t)</td>
          <td>0.145934</td>
          <td>0.99</td>
        </tr>
        <tr>
          <th>19</th>
          <td>x0(t-1)</td>
          <td>x2(t)</td>
          <td>0.109297</td>
          <td>0.98</td>
        </tr>
        <tr>
          <th>20</th>
          <td>x3(t-1)</td>
          <td>x2(t)</td>
          <td>-0.113304</td>
          <td>0.98</td>
        </tr>
        <tr>
          <th>21</th>
          <td>x4(t-1)</td>
          <td>x0(t)</td>
          <td>-0.055275</td>
          <td>0.95</td>
        </tr>
        <tr>
          <th>22</th>
          <td>x1(t-1)</td>
          <td>x2(t)</td>
          <td>-0.048436</td>
          <td>0.95</td>
        </tr>
        <tr>
          <th>23</th>
          <td>x0(t-1)</td>
          <td>x4(t)</td>
          <td>-0.052491</td>
          <td>0.93</td>
        </tr>
        <tr>
          <th>24</th>
          <td>x1(t)</td>
          <td>x4(t)</td>
          <td>-0.038710</td>
          <td>0.92</td>
        </tr>
        <tr>
          <th>25</th>
          <td>x0(t-1)</td>
          <td>x1(t)</td>
          <td>0.032712</td>
          <td>0.90</td>
        </tr>
        <tr>
          <th>26</th>
          <td>x1(t-1)</td>
          <td>x0(t)</td>
          <td>0.026323</td>
          <td>0.83</td>
        </tr>
        <tr>
          <th>27</th>
          <td>x2(t-1)</td>
          <td>x4(t)</td>
          <td>-0.003520</td>
          <td>0.81</td>
        </tr>
        <tr>
          <th>28</th>
          <td>x4(t-1)</td>
          <td>x4(t)</td>
          <td>-0.020322</td>
          <td>0.78</td>
        </tr>
        <tr>
          <th>29</th>
          <td>x3(t)</td>
          <td>x4(t)</td>
          <td>-0.074582</td>
          <td>0.70</td>
        </tr>
        <tr>
          <th>30</th>
          <td>x0(t)</td>
          <td>x3(t)</td>
          <td>0.077178</td>
          <td>0.69</td>
        </tr>
        <tr>
          <th>31</th>
          <td>x2(t)</td>
          <td>x4(t)</td>
          <td>-0.064105</td>
          <td>0.67</td>
        </tr>
        <tr>
          <th>32</th>
          <td>x1(t-1)</td>
          <td>x3(t)</td>
          <td>-0.000250</td>
          <td>0.59</td>
        </tr>
        <tr>
          <th>33</th>
          <td>x1(t-1)</td>
          <td>x4(t)</td>
          <td>0.002664</td>
          <td>0.56</td>
        </tr>
        <tr>
          <th>34</th>
          <td>x3(t)</td>
          <td>x2(t)</td>
          <td>0.008626</td>
          <td>0.50</td>
        </tr>
        <tr>
          <th>35</th>
          <td>x4(t)</td>
          <td>x2(t)</td>
          <td>-0.062254</td>
          <td>0.33</td>
        </tr>
        <tr>
          <th>36</th>
          <td>x2(t)</td>
          <td>x3(t)</td>
          <td>0.006647</td>
          <td>0.32</td>
        </tr>
        <tr>
          <th>37</th>
          <td>x3(t)</td>
          <td>x0(t)</td>
          <td>0.057305</td>
          <td>0.29</td>
        </tr>
        <tr>
          <th>38</th>
          <td>x4(t)</td>
          <td>x3(t)</td>
          <td>-0.040263</td>
          <td>0.27</td>
        </tr>
        <tr>
          <th>39</th>
          <td>x4(t)</td>
          <td>x0(t)</td>
          <td>0.081813</td>
          <td>0.01</td>
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
          <td>0.496761</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>9</th>
          <td>x4(t-1)</td>
          <td>x1(t)</td>
          <td>0.454093</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>11</th>
          <td>x2(t-1)</td>
          <td>x2(t)</td>
          <td>0.354316</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>13</th>
          <td>x3(t-1)</td>
          <td>x0(t)</td>
          <td>0.339193</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>18</th>
          <td>x0(t)</td>
          <td>x4(t)</td>
          <td>0.145934</td>
          <td>0.99</td>
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
          <th>9</th>
          <td>x4(t-1)</td>
          <td>x1(t)</td>
          <td>0.454093</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>10</th>
          <td>x3(t-1)</td>
          <td>x1(t)</td>
          <td>-0.353886</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>12</th>
          <td>x1(t-1)</td>
          <td>x1(t)</td>
          <td>-0.294882</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>15</th>
          <td>x2(t-1)</td>
          <td>x1(t)</td>
          <td>-0.192527</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>25</th>
          <td>x0(t-1)</td>
          <td>x1(t)</td>
          <td>0.032712</td>
          <td>0.9</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br>



Because it holds the raw data of the causal effect (the original data for calculating the median), it is possible to draw a histogram of the values of the causal effect, as shown below.

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    %matplotlib inline
    
    from_index = 7 # index of x2(t-1). (index:2)+(n_features:5)*(lag:1) = 7
    to_index = 2 # index of x2(t). (index:2)+(n_features:5)*(lag:0) = 2
    plt.hist(result.total_effects_[:, to_index, from_index])


.. image:: ../image/var_hist.png


