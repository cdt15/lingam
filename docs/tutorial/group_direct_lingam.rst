GroupDirectLiNGAM
=================

Import and settings
-------------------

In this example, we need to import ``numpy``, ``pandas``, and
``graphviz`` in addition to ``lingam``.

.. code:: ipython3

    import numpy as np
    import pandas as pd
    import graphviz
    import lingam
    from lingam.utils import print_causal_directions, print_dagc, make_prior_knowledge, make_dot

    import warnings
    warnings.filterwarnings('ignore')

    print([np.__version__, pd.__version__, graphviz.__version__, lingam.__version__])

    np.set_printoptions(precision=3, suppress=True)


.. parsed-literal::

    ['1.26.4', '2.2.3', '0.20.3', '1.11.0']


Test data
---------

First, we generate a causal structure with 6 variables. Then we create a
dataset with 6 variables from x0 to x5.

These variables are grouped as follows: - Group 1: x0, x1 - Group 2: x2
- Group 3: x3, x4 - Group 4: x5

.. code:: ipython3

    np.random.seed(0)

    n_samples = 1000
    x0 = np.random.uniform(size=n_samples)
    x1 = 2.0 * x0 + np.random.uniform(size=n_samples)
    x2 = 0.5 * x1 + np.random.uniform(-1, 1, size=n_samples)
    x3 = 0.3 * x1 + 0.7 * x2 + np.random.uniform(-2, 2, size=n_samples)
    x4 = 1.5 * x0 + 0.8 * x3 + np.random.uniform(-2, 2, size=n_samples)
    x5 = -0.6 * x3 - 0.5 * x4 + np.random.uniform(-3, 3, size=n_samples)

    X = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T ,columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])
    groups = [[0, 1], [2], [3, 4], [5]]

    X.head()




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
          <td>0.548814</td>
          <td>1.690507</td>
          <td>1.468291</td>
          <td>1.190806</td>
          <td>0.946433</td>
          <td>-1.987018</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.715189</td>
          <td>1.440442</td>
          <td>0.672389</td>
          <td>1.421278</td>
          <td>2.475880</td>
          <td>-3.304962</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.602763</td>
          <td>1.681353</td>
          <td>0.886988</td>
          <td>2.239635</td>
          <td>1.245511</td>
          <td>-4.554939</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.544883</td>
          <td>1.798537</td>
          <td>0.400310</td>
          <td>2.226009</td>
          <td>1.996981</td>
          <td>-3.218930</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.423655</td>
          <td>0.891285</td>
          <td>0.655729</td>
          <td>1.992046</td>
          <td>0.441985</td>
          <td>-3.023044</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    m = np.array([
        [  0,   0,   0,   0,   0, 0],
        [2.0,   0,   0,   0,   0, 0],
        [  0, 0.5,   0,   0,   0, 0],
        [  0, 0.3, 0.7,   0,   0, 0],
        [1.5,   0,   0, 0.8,   0, 0],
        [  0,   0,   0,-0.6,-0.5, 0]])

    dot = make_dot(m, labels=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])

    # Save pdf
    dot.render('dag')

    # Save png
    dot.format = 'png'
    dot.render('dag')

    dot




.. image:: ../image/group_lingam.svg



Causal Discovery
----------------

To run causal discovery, we create a ``GroupDirectLiNGAM`` object and
call the ``fit`` method.

.. code:: ipython3

    model = lingam.GroupDirectLiNGAM()
    model.fit(X, groups)




.. parsed-literal::

    <lingam.group_direct_lingam.GroupDirectLiNGAM at 0x1d17f5af890>



Using the ``causal_order_`` properties, we can see the causal order of
the groups as a result of the causal discovery.

.. code:: ipython3

    model.causal_order_




.. parsed-literal::

    [0, 1, 2, 3]



The causal order of the variables is as follows:

.. code:: ipython3

    [groups[group_idx] for group_idx in model.causal_order_]




.. parsed-literal::

    [[0, 1], [2], [3, 4], [5]]



Also, using the ``adjacency_matrix_`` properties, we can see the
adjacency matrix as a result of the causal discovery.

.. code:: ipython3

    model.adjacency_matrix_




.. parsed-literal::

    array([[ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
           [ 0.   ,  0.482,  0.   ,  0.   ,  0.   ,  0.   ],
           [ 0.   ,  0.194,  0.792,  0.   ,  0.   ,  0.   ],
           [ 1.882,  0.   ,  0.572,  0.   ,  0.   ,  0.   ],
           [ 0.   ,  0.   ,  0.   , -0.622, -0.487,  0.   ]])



.. code:: ipython3

    make_dot(model.adjacency_matrix_)




.. image:: ../image/group_lingam2.svg



Bootstrapping
-------------

We call ``bootstrap()`` method instead of ``fit()``. Here, the third
argument specifies the number of bootstrap sampling.

.. code:: ipython3

    model = lingam.GroupDirectLiNGAM()
    result = model.bootstrap(X, groups, 100)

Causal Directions
-----------------

Since ``BootstrapResult`` object is returned, we can get the ranking of
the causal directions extracted by ``get_causal_direction_counts()``
method. In the following sample code, ``n_directions`` option is limited
to the causal directions of the top 8 rankings, and
``min_causal_effect`` option is limited to causal directions with a
coefficient of 0.01 or more.

.. code:: ipython3

    cdc = result.get_causal_direction_counts(n_directions=8, min_causal_effect=0.01, split_by_causal_effect_sign=True)

We can check the result by utility function.

.. code:: ipython3

    print_causal_directions(cdc, 100)


.. parsed-literal::

    x2 <--- x1 (b>0) (100.0%)
    x3 <--- x2 (b>0) (100.0%)
    x4 <--- x2 (b>0) (100.0%)
    x5 <--- x3 (b<0) (100.0%)
    x5 <--- x4 (b<0) (100.0%)
    x4 <--- x0 (b>0) (99.0%)
    x3 <--- x1 (b>0) (49.0%)
    x3 <--- x0 (b>0) (23.0%)


Directed Acyclic Graphs
-----------------------

Also, using the ``get_directed_acyclic_graph_counts()`` method, we can
get the ranking of the DAGs extracted. In the following sample code,
``n_dags`` option is limited to the dags of the top 3 rankings, and
``min_causal_effect`` option is limited to causal directions with a
coefficient of 0.01 or more.

.. code:: ipython3

    dagc = result.get_directed_acyclic_graph_counts(n_dags=3, min_causal_effect=0.01, split_by_causal_effect_sign=True)

We can check the result by utility function.

.. code:: ipython3

    print_dagc(dagc, 100)


.. parsed-literal::

    DAG[0]: 35.0%
    	x2 <--- x1 (b>0)
    	x3 <--- x1 (b>0)
    	x3 <--- x2 (b>0)
    	x4 <--- x0 (b>0)
    	x4 <--- x2 (b>0)
    	x5 <--- x3 (b<0)
    	x5 <--- x4 (b<0)
    DAG[1]: 27.0%
    	x2 <--- x1 (b>0)
    	x3 <--- x2 (b>0)
    	x4 <--- x0 (b>0)
    	x4 <--- x2 (b>0)
    	x5 <--- x3 (b<0)
    	x5 <--- x4 (b<0)
    DAG[2]: 19.0%
    	x2 <--- x1 (b>0)
    	x3 <--- x0 (b>0)
    	x3 <--- x2 (b>0)
    	x4 <--- x0 (b>0)
    	x4 <--- x2 (b>0)
    	x5 <--- x3 (b<0)
    	x5 <--- x4 (b<0)


Probability
-----------

Using the ``get_probabilities()`` method, we can get the probability of
bootstrapping.

.. code:: ipython3

    prob = result.get_probabilities(min_causal_effect=0.01)
    print(prob)


.. parsed-literal::

    [[0.   0.   0.   0.   0.   0.  ]
     [0.   0.   0.   0.   0.   0.  ]
     [0.08 1.   0.   0.   0.   0.  ]
     [0.23 0.49 1.   0.   0.   0.  ]
     [0.99 0.07 1.   0.   0.   0.  ]
     [0.01 0.03 0.   1.   1.   0.  ]]


Total Causal Effects
--------------------

Using the ``get_total_causal_effects()`` method, we can get the list of
total causal effect. The total causal effects we can get are dictionary
type variable. We can display the list nicely by assigning it to
pandas.DataFrame. Also, we have replaced the variable index with a label
below.

.. code:: ipython3

    causal_effects = result.get_total_causal_effects(min_causal_effect=0.01)

    # Assign to pandas.DataFrame for pretty display
    df = pd.DataFrame(causal_effects)
    labels = [f'x{i}' for i in range(X.shape[1])]
    df['from'] = df['from'].apply(lambda x : labels[x])
    df['to'] = df['to'].apply(lambda x : labels[x])
    df




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
          <th>from</th>
          <th>to</th>
          <th>effect</th>
          <th>probability</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>x1</td>
          <td>x2</td>
          <td>0.483013</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>1</th>
          <td>x1</td>
          <td>x3</td>
          <td>0.504680</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>2</th>
          <td>x2</td>
          <td>x3</td>
          <td>0.813637</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>3</th>
          <td>x1</td>
          <td>x4</td>
          <td>0.278019</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>4</th>
          <td>x2</td>
          <td>x4</td>
          <td>0.555515</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>5</th>
          <td>x1</td>
          <td>x5</td>
          <td>-0.456973</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>6</th>
          <td>x2</td>
          <td>x5</td>
          <td>-0.786690</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>7</th>
          <td>x3</td>
          <td>x5</td>
          <td>-0.629688</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>8</th>
          <td>x4</td>
          <td>x5</td>
          <td>-0.480164</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>9</th>
          <td>x0</td>
          <td>x4</td>
          <td>1.855977</td>
          <td>0.99</td>
        </tr>
        <tr>
          <th>10</th>
          <td>x0</td>
          <td>x5</td>
          <td>-0.913577</td>
          <td>0.99</td>
        </tr>
        <tr>
          <th>11</th>
          <td>x0</td>
          <td>x3</td>
          <td>0.385821</td>
          <td>0.29</td>
        </tr>
        <tr>
          <th>12</th>
          <td>x0</td>
          <td>x2</td>
          <td>-0.395791</td>
          <td>0.08</td>
        </tr>
      </tbody>
    </table>
    </div>



We can easily perform sorting operations with pandas.DataFrame.

.. code:: ipython3

    df.sort_values('effect', ascending=False).head()




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
          <th>from</th>
          <th>to</th>
          <th>effect</th>
          <th>probability</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>9</th>
          <td>x0</td>
          <td>x4</td>
          <td>1.855977</td>
          <td>0.99</td>
        </tr>
        <tr>
          <th>2</th>
          <td>x2</td>
          <td>x3</td>
          <td>0.813637</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>4</th>
          <td>x2</td>
          <td>x4</td>
          <td>0.555515</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>1</th>
          <td>x1</td>
          <td>x3</td>
          <td>0.504680</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>0</th>
          <td>x1</td>
          <td>x2</td>
          <td>0.483013</td>
          <td>1.00</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    df.sort_values('probability', ascending=True).head()




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
          <th>from</th>
          <th>to</th>
          <th>effect</th>
          <th>probability</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>12</th>
          <td>x0</td>
          <td>x2</td>
          <td>-0.395791</td>
          <td>0.08</td>
        </tr>
        <tr>
          <th>11</th>
          <td>x0</td>
          <td>x3</td>
          <td>0.385821</td>
          <td>0.29</td>
        </tr>
        <tr>
          <th>9</th>
          <td>x0</td>
          <td>x4</td>
          <td>1.855977</td>
          <td>0.99</td>
        </tr>
        <tr>
          <th>10</th>
          <td>x0</td>
          <td>x5</td>
          <td>-0.913577</td>
          <td>0.99</td>
        </tr>
        <tr>
          <th>0</th>
          <td>x1</td>
          <td>x2</td>
          <td>0.483013</td>
          <td>1.00</td>
        </tr>
      </tbody>
    </table>
    </div>



Because it holds the raw data of the total causal effect (the original
data for calculating the median), it is possible to draw a histogram of
the values of the causal effect, as shown below.

.. code:: ipython3

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    %matplotlib inline

    from_index = 0 # index of x0
    to_index = 5 # index of x5
    plt.hist(result.total_effects_[:, to_index, from_index])




.. parsed-literal::

    (array([ 1.,  0.,  0.,  0.,  8., 25., 48., 13.,  4.,  1.]),
     array([-2.527, -2.274, -2.021, -1.769, -1.516, -1.263, -1.011, -0.758,
            -0.505, -0.253,  0.   ]),
     <BarContainer object of 10 artists>)




.. image:: ../image/group_lingam3.png


Bootstrap Probability of Path
-----------------------------

Using the ``get_paths()`` method, we can explore all paths from any
variable to any variable and calculate the bootstrap probability for
each path. The path will be output as an array of variable indices. For
example, the array ``[3, 0, 1]`` shows the path from variable X3 through
variable X0 to variable X1.

.. code:: ipython3

    from_index = 0 # index of x0
    to_index = 5 # index of x5

    pd.DataFrame(result.get_paths(from_index, to_index))




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
          <td>[0, 4, 5]</td>
          <td>-0.899122</td>
          <td>0.99</td>
        </tr>
        <tr>
          <th>1</th>
          <td>[0, 3, 5]</td>
          <td>-0.280465</td>
          <td>0.23</td>
        </tr>
        <tr>
          <th>2</th>
          <td>[0, 2, 3, 5]</td>
          <td>0.181984</td>
          <td>0.08</td>
        </tr>
        <tr>
          <th>3</th>
          <td>[0, 2, 4, 5]</td>
          <td>0.118028</td>
          <td>0.08</td>
        </tr>
        <tr>
          <th>4</th>
          <td>[0, 5]</td>
          <td>-1.663298</td>
          <td>0.01</td>
        </tr>
      </tbody>
    </table>
    </div>



