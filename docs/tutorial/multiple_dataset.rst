
MultiGroupDirectLiNGAM
======================

Model
-------------------
This algorithm [3]_ simultaneously analyzes multiple datasets obtained from different sources, e.g., from groups of different ages.  
The algorithm is an extention of DirectLiNGAM [1]_ to multiple-group cases.
The algorithm assumes that each dataset comes from a basic LiNGAM model [2]_, i.e., makes the following assumptions in each dataset:

#. Linearity
#. Non-Gaussian continuous error variables (except at most one)
#. Acyclicity
#. No hidden common causes

Further, it assumes the topological causal orders are common to the groups. 
The similarity in the topological causal orders would give a better performance than analyzing each dataset separatly if the assumption on the causal orders are reasonable. 

Denote observed variables of Group :math:`{g}` by :math:`{x}_{i}^{(g)}` and error variables by :math:`{e}_{i}^{(g)}` and coefficients or connection strengths :math:`{b}_{ij}^{(g)}`. 
Collect them in vectors :math:`{x}`and :math:`{e}^{(g)}` and a matrix :math:`{B}^{(g)}`, respectivelly. 
Due to the assumptions of acyclicity and common topological causal orders, the adjacency matrix :math:`{B}^{(g)}` can be permuted to be strictly lower-triangular by a simultaneous row and column permutation common to the groups. 
The error variables :math:`{e}_{i}^{(g)}` are independent due to the assumption of no hidden common causes. 

Then, mathematically, the model for observed variable vector :math:`{x}^{(g)}` is written as 

$$ x^{(g)} = B^{(g)}x^{(g)} + e^{(g)}. $$

References

    .. [1] S. Shimizu, T. Inazumi, Y. Sogawa, A. Hyvärinen, Y. Kawahara, T. Washio, P. O. Hoyer and K. Bollen. 
        DirectLiNGAM: A direct method for learning a linear non-Gaussian structural equation model. 
        Journal of Machine Learning Research, 12(Apr): 1225–1248, 2011.
    .. [2] S. Shimizu, P. O. Hoyer, A. Hyvärinen, and A. J. Kerminen.
       A linear non-gaussian acyclic model for causal discovery.
       Journal of Machine Learning Research, 7:2003-2030, 2006.
    .. [3] S. Shimizu. Joint estimation of linear non-Gaussian acyclic models. 
        Neurocomputing, 81: 104-107, 2012.

Import and settings
-------------------

In this example, we need to import ``numpy``, ``pandas``, and
``graphviz`` in addition to ``lingam``.

.. code-block:: python

    import numpy as np
    import pandas as pd
    import graphviz
    import lingam
    from lingam.utils import print_causal_directions, print_dagc, make_dot
    
    print([np.__version__, pd.__version__, graphviz.__version__, lingam.__version__])
    
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(0)


.. parsed-literal::

    ['1.24.4', '2.0.3', '0.20.1', '1.8.3']


Test data
---------

We generate two datasets consisting of 6 variables.

.. code-block:: python

    x3 = np.random.uniform(size=1000)
    x0 = 3.0*x3 + np.random.uniform(size=1000)
    x2 = 6.0*x3 + np.random.uniform(size=1000)
    x1 = 3.0*x0 + 2.0*x2 + np.random.uniform(size=1000)
    x5 = 4.0*x0 + np.random.uniform(size=1000)
    x4 = 8.0*x0 - 1.0*x2 + np.random.uniform(size=1000)
    X1 = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T ,columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])
    X1.head()




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
          <th>x0</th>
          <th>x1</th>
          <th>x2</th>
          <th>x3</th>
          <th>x4</th>
          <th>x5</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>2.239321</td>
          <td>15.340724</td>
          <td>4.104399</td>
          <td>0.548814</td>
          <td>14.176947</td>
          <td>9.249925</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.155632</td>
          <td>16.630954</td>
          <td>4.767220</td>
          <td>0.715189</td>
          <td>12.775458</td>
          <td>9.189045</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2.284116</td>
          <td>15.910406</td>
          <td>4.139736</td>
          <td>0.602763</td>
          <td>14.201794</td>
          <td>9.273880</td>
        </tr>
        <tr>
          <th>3</th>
          <td>2.343420</td>
          <td>14.921457</td>
          <td>3.519820</td>
          <td>0.544883</td>
          <td>15.580067</td>
          <td>9.723392</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.314940</td>
          <td>11.055176</td>
          <td>3.146972</td>
          <td>0.423655</td>
          <td>7.604743</td>
          <td>5.312976</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code-block:: python

    m = np.array([[0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                  [3.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [8.0, 0.0,-1.0, 0.0, 0.0, 0.0],
                  [4.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    
    make_dot(m)




.. image:: ../image/multiple_dataset_dag1.svg



.. code-block:: python

    x3 = np.random.uniform(size=1000)
    x0 = 3.5*x3 + np.random.uniform(size=1000)
    x2 = 6.5*x3 + np.random.uniform(size=1000)
    x1 = 3.5*x0 + 2.5*x2 + np.random.uniform(size=1000)
    x5 = 4.5*x0 + np.random.uniform(size=1000)
    x4 = 8.5*x0 - 1.5*x2 + np.random.uniform(size=1000)
    X2 = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T ,columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])
    X2.head()




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
          <th>x0</th>
          <th>x1</th>
          <th>x2</th>
          <th>x3</th>
          <th>x4</th>
          <th>x5</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.913337</td>
          <td>14.568170</td>
          <td>2.893918</td>
          <td>0.374794</td>
          <td>12.115455</td>
          <td>9.358286</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.013935</td>
          <td>15.857260</td>
          <td>3.163377</td>
          <td>0.428686</td>
          <td>12.657021</td>
          <td>9.242911</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3.172835</td>
          <td>24.734385</td>
          <td>5.142203</td>
          <td>0.683057</td>
          <td>19.605722</td>
          <td>14.666783</td>
        </tr>
        <tr>
          <th>3</th>
          <td>2.990395</td>
          <td>20.878961</td>
          <td>4.113485</td>
          <td>0.600948</td>
          <td>19.452091</td>
          <td>13.494380</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.248702</td>
          <td>2.268163</td>
          <td>0.532419</td>
          <td>0.070483</td>
          <td>1.854870</td>
          <td>1.130948</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br>



.. code-block:: python

    m = np.array([[0.0, 0.0, 0.0, 3.5, 0.0, 0.0],
                  [3.5, 0.0, 2.5, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 6.5, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [8.5, 0.0,-1.5, 0.0, 0.0, 0.0],
                  [4.5, 0.0, 0.0, 0.0, 0.0, 0.0]])
    
    make_dot(m)




.. image:: ../image/multiple_dataset_dag2.svg



We create a list variable that contains two datasets.

.. code-block:: python

    X_list = [X1, X2]

Causal Discovery
----------------

To run causal discovery for multiple datasets, we create a :class:`~lingam.MultiGroupDirectLiNGAM` object and call the :func:`~lingam.MultiGroupDirectLiNGAM.fit` method.

.. code-block:: python

    model = lingam.MultiGroupDirectLiNGAM()
    model.fit(X_list)




.. parsed-literal::

    <lingam.multi_group_direct_lingam.MultiGroupDirectLiNGAM at 0x7f87aca0fcd0>



Using the :attr:`~lingam.MultiGroupDirectLiNGAM.causal_order_` properties, we can see the causal ordering as a result of the causal discovery.

.. code-block:: python

    model.causal_order_




.. parsed-literal::

    [3, 0, 5, 2, 1, 4]



Also, using the :attr:`~lingam.MultiGroupDirectLiNGAM.adjacency_matrix_` properties, we can see the adjacency matrix as a result of the causal discovery. As you can see from the following, DAG in each dataset is correctly estimated.

.. code-block:: python

    print(model.adjacency_matrices_[0])
    make_dot(model.adjacency_matrices_[0])


.. parsed-literal::

    [[ 0.     0.     0.     3.006  0.     0.   ]
     [ 2.962  0.     2.015  0.     0.     0.   ]
     [ 0.     0.     0.     5.97   0.     0.   ]
     [ 0.     0.     0.     0.     0.     0.   ]
     [ 8.01   0.    -1.006  0.     0.     0.   ]
     [ 3.998  0.     0.     0.     0.     0.   ]]



.. image:: ../image/multiple_dataset_dag3.svg



.. code-block:: python

    print(model.adjacency_matrices_[1])
    make_dot(model.adjacency_matrices_[1])


.. parsed-literal::

    [[ 0.     0.     0.     3.483  0.     0.   ]
     [ 3.526  0.     2.486  0.     0.     0.   ]
     [ 0.     0.     0.     6.503  0.     0.   ]
     [ 0.     0.     0.     0.     0.     0.   ]
     [ 8.478  0.    -1.484  0.     0.     0.   ]
     [ 4.494  0.     0.     0.     0.     0.   ]]




.. image:: ../image/multiple_dataset_dag4.svg



To compare, we run DirectLiNGAM with single dataset concatenating two
datasets.

.. code-block:: python

    X_all = pd.concat([X1, X2])
    print(X_all.shape)


.. parsed-literal::

    (2000, 6)
    

.. code-block:: python

    model_all = lingam.DirectLiNGAM()
    model_all.fit(X_all)
    
    model_all.causal_order_




.. parsed-literal::

    [1, 5, 2, 3, 0, 4]



You can see that the causal structure cannot be estimated correctly for
a single dataset.

.. code-block:: python

    make_dot(model_all.adjacency_matrix_)




.. image:: ../image/multiple_dataset_dag5.svg



Independence between error variables
------------------------------------

To check if the LiNGAM assumption is broken, we can get p-values of
independence between error variables. The value in the i-th row and j-th
column of the obtained matrix shows the p-value of the independence of
the error variables :math:`e_i` and :math:`e_j`.

.. code-block:: python

    p_values = model.get_error_independence_p_values(X_list)
    print(p_values[0])


.. parsed-literal::

    [[0.    0.939 0.07  0.838 0.467 0.818]
     [0.939 0.    0.195 0.755 0.053 0.254]
     [0.07  0.195 0.    0.851 0.374 0.507]
     [0.838 0.755 0.851 0.    0.422 0.755]
     [0.467 0.053 0.374 0.422 0.    0.581]
     [0.818 0.254 0.507 0.755 0.581 0.   ]]

.. code-block:: python

    print(p_values[1])


.. parsed-literal::

    [[0.    0.521 0.973 0.285 0.459 0.606]
     [0.521 0.    0.805 0.785 0.078 0.395]
     [0.973 0.805 0.    0.807 0.378 0.129]
     [0.285 0.785 0.807 0.    0.913 0.99 ]
     [0.459 0.078 0.378 0.913 0.    0.269]
     [0.606 0.395 0.129 0.99  0.269 0.   ]]


Bootstrapping
-------------

In :class:`~lingam.MultiGroupDirectLiNGAM`, bootstrap can be executed in the same way as normal :class:`~lingam.DirectLiNGAM`.

.. code-block:: python

    results = model.bootstrap(X_list, n_sampling=100)

Causal Directions
-----------------

The :func:`~lingam.MultiGroupDirectLiNGAM.bootstrap` method returns a list of multiple :class:`~lingam.BootstrapResult`, so we can get the result of bootstrapping from the list. We can get the same number of results as the number of datasets, so we specify an index when we access the results. We can get the ranking of the causal directions extracted by :func:`~lingam.BootstrapResult.get_causal_direction_counts`.

.. code-block:: python

    cdc = results[0].get_causal_direction_counts(n_directions=8, min_causal_effect=0.01)
    print_causal_directions(cdc, 100)


.. parsed-literal::

    x0 <--- x3  (100.0%)
    x1 <--- x0  (100.0%)
    x1 <--- x2  (100.0%)
    x2 <--- x3  (100.0%)
    x4 <--- x0  (100.0%)
    x4 <--- x2  (100.0%)
    x5 <--- x0  (100.0%)
    x1 <--- x3  (4.0%)

.. code-block:: python

    cdc = results[1].get_causal_direction_counts(n_directions=8, min_causal_effect=0.01)
    print_causal_directions(cdc, 100)


.. parsed-literal::

    x0 <--- x3  (100.0%)
    x1 <--- x0  (100.0%)
    x1 <--- x2  (100.0%)
    x5 <--- x0  (100.0%)
    x2 <--- x3  (100.0%)
    x4 <--- x0  (100.0%)
    x4 <--- x2  (100.0%)
    x4 <--- x3  (12.0%)


Directed Acyclic Graphs
-----------------------

Also, using the :func:`~lingam.BootstrapResult.get_directed_acyclic_graph_counts` method, we can get the ranking of the DAGs extracted. In the following sample code, ``n_dags`` option is limited to the dags of the top 3 rankings, and ``min_causal_effect`` option is limited to causal directions with a coefficient of 0.01 or more.

.. code-block:: python

    dagc = results[0].get_directed_acyclic_graph_counts(n_dags=3, min_causal_effect=0.01)
    print_dagc(dagc, 100)


.. parsed-literal::

    DAG[0]: 87.0%
    	x0 <--- x3 
    	x1 <--- x0 
    	x1 <--- x2 
    	x2 <--- x3 
    	x4 <--- x0 
    	x4 <--- x2 
    	x5 <--- x0 
    DAG[1]: 4.0%
    	x0 <--- x3 
    	x1 <--- x0 
    	x1 <--- x2 
    	x2 <--- x3 
    	x4 <--- x0 
    	x4 <--- x2 
    	x5 <--- x0 
    	x5 <--- x2 
    	x5 <--- x3 
    DAG[2]: 4.0%
    	x0 <--- x3 
    	x1 <--- x0 
    	x1 <--- x2 
    	x1 <--- x3 
    	x2 <--- x3 
    	x4 <--- x0 
    	x4 <--- x2 
    	x5 <--- x0 

.. code-block:: python

    dagc = results[1].get_directed_acyclic_graph_counts(n_dags=3, min_causal_effect=0.01)
    print_dagc(dagc, 100)


.. parsed-literal::

    DAG[0]: 60.0%
    	x0 <--- x3 
    	x1 <--- x0 
    	x1 <--- x2 
    	x2 <--- x3 
    	x4 <--- x0 
    	x4 <--- x2 
    	x5 <--- x0 
    DAG[1]: 8.0%
    	x0 <--- x3 
    	x1 <--- x0 
    	x1 <--- x2 
    	x2 <--- x3 
    	x4 <--- x0 
    	x4 <--- x2 
    	x5 <--- x0 
    	x5 <--- x2 
    DAG[2]: 7.0%
    	x0 <--- x3 
    	x1 <--- x0 
    	x1 <--- x2 
    	x2 <--- x3 
    	x4 <--- x0 
    	x4 <--- x2 
    	x4 <--- x3 
    	x5 <--- x0 


Probability
-----------

Using the :func:`~lingam.BootstrapResult.get_probabilities` method, we can get the probability of bootstrapping.

.. code-block:: python

    prob = results[0].get_probabilities(min_causal_effect=0.01)
    print(prob)


.. parsed-literal::

    [[0.   0.   0.   1.   0.   0.  ]
     [1.   0.   1.   0.04 0.   0.02]
     [0.01 0.   0.   1.   0.   0.  ]
     [0.   0.   0.   0.   0.   0.  ]
     [1.   0.   1.   0.02 0.   0.  ]
     [1.   0.   0.04 0.04 0.   0.  ]]


Total Causal Effects
--------------------

Using the ``get_total_causal_effects()`` method, we can get the list of
total causal effect. The total causal effects we can get are dictionary
type variable. We can display the list nicely by assigning it to
pandas.DataFrame. Also, we have replaced the variable index with a label
below.

.. code-block:: python

    causal_effects = results[0].get_total_causal_effects(min_causal_effect=0.01)
    df = pd.DataFrame(causal_effects)
    
    labels = [f'x{i}' for i in range(X1.shape[1])]
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
          <td>x3</td>
          <td>x0</td>
          <td>3.005604</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>1</th>
          <td>x0</td>
          <td>x1</td>
          <td>2.963048</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>2</th>
          <td>x2</td>
          <td>x1</td>
          <td>2.016516</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>3</th>
          <td>x3</td>
          <td>x1</td>
          <td>20.943993</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>4</th>
          <td>x3</td>
          <td>x2</td>
          <td>5.969457</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>5</th>
          <td>x0</td>
          <td>x4</td>
          <td>8.011896</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>6</th>
          <td>x2</td>
          <td>x4</td>
          <td>-1.007165</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>7</th>
          <td>x3</td>
          <td>x4</td>
          <td>18.061596</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>8</th>
          <td>x0</td>
          <td>x5</td>
          <td>4.001221</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>9</th>
          <td>x3</td>
          <td>x5</td>
          <td>12.026086</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>10</th>
          <td>x2</td>
          <td>x5</td>
          <td>-0.119097</td>
          <td>0.04</td>
        </tr>
        <tr>
          <th>11</th>
          <td>x5</td>
          <td>x1</td>
          <td>0.092263</td>
          <td>0.02</td>
        </tr>
        <tr>
          <th>12</th>
          <td>x0</td>
          <td>x2</td>
          <td>0.079547</td>
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
          <th>3</th>
          <td>x3</td>
          <td>x1</td>
          <td>20.943993</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>7</th>
          <td>x3</td>
          <td>x4</td>
          <td>18.061596</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>9</th>
          <td>x3</td>
          <td>x5</td>
          <td>12.026086</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>5</th>
          <td>x0</td>
          <td>x4</td>
          <td>8.011896</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>x3</td>
          <td>x2</td>
          <td>5.969457</td>
          <td>1.0</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br>



And with pandas.DataFrame, we can easily filter by keywords. The
following code extracts the causal direction towards x1.

.. code-block:: python

    df[df['to']=='x1'].head()




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
          <th>1</th>
          <td>x0</td>
          <td>x1</td>
          <td>2.963048</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>2</th>
          <td>x2</td>
          <td>x1</td>
          <td>2.016516</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>3</th>
          <td>x3</td>
          <td>x1</td>
          <td>20.943993</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>11</th>
          <td>x5</td>
          <td>x1</td>
          <td>0.092263</td>
          <td>0.02</td>
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
    
    from_index = 3
    to_index = 0
    plt.hist(results[0].total_effects_[:, to_index, from_index])


Bootstrap Probability of Path
-----------------------------

Using the ``get_paths()`` method, we can explore all paths from any
variable to any variable and calculate the bootstrap probability for
each path. The path will be output as an array of variable indices. For
example, the array ``[3, 0, 1]`` shows the path from variable X3 through
variable X0 to variable X1.

.. code-block:: python

    from_index = 3 # index of x3
    to_index = 1 # index of x0
    
    pd.DataFrame(results[0].get_paths(from_index, to_index))




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>path</th>
          <th>effect</th>
          <th>probability</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>[3, 0, 1]</td>
          <td>8.905047</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>1</th>
          <td>[3, 2, 1]</td>
          <td>12.049228</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>2</th>
          <td>[3, 1]</td>
          <td>-0.733971</td>
          <td>0.04</td>
        </tr>
        <tr>
          <th>3</th>
          <td>[3, 0, 5, 1]</td>
          <td>1.096850</td>
          <td>0.02</td>
        </tr>
        <tr>
          <th>4</th>
          <td>[3, 0, 2, 1]</td>
          <td>0.493177</td>
          <td>0.01</td>
        </tr>
      </tbody>
    </table>
    </div>



