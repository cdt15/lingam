BottomUpParceLiNGAM
===================

Import and settings
-------------------

In this example, we need to import ``numpy``, ``pandas``, and
``graphviz`` in addition to ``lingam``.

.. code:: ipython3

    import numpy as np
    import pandas as pd
    import graphviz
    import lingam
    from lingam.utils import print_causal_directions, print_dagc, make_dot

    print([np.__version__, pd.__version__, graphviz.__version__, lingam.__version__])

    np.set_printoptions(precision=3, suppress=True)


.. parsed-literal::

    ['1.18.1', '1.0.1', '0.14', '1.4.0']


Test data
---------

First, we generate a causal structure with 7 variables. Then we create a
dataset with 6 variables from x0 to x5, with x6 being the latent
variable for x2 and x3.

.. code:: ipython3

    np.random.seed(1000)

    x6 = np.random.uniform(size=1000)
    x3 = 2.0*x6 + np.random.uniform(size=1000)
    x0 = 0.5*x3 + np.random.uniform(size=1000)
    x2 = 2.0*x6 + np.random.uniform(size=1000)
    x1 = 0.5*x0 + 0.5*x2 + np.random.uniform(size=1000)
    x5 = 0.5*x0 + np.random.uniform(size=1000)
    x4 = 0.5*x0 - 0.5*x2 + np.random.uniform(size=1000)

    # The latent variable x6 is not included.
    X = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T, columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])

    X.head()




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
          <td>1.505949</td>
          <td>2.667827</td>
          <td>2.029420</td>
          <td>1.463708</td>
          <td>0.615387</td>
          <td>1.157907</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.379130</td>
          <td>1.721744</td>
          <td>0.965613</td>
          <td>0.801952</td>
          <td>0.919654</td>
          <td>0.957148</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.436825</td>
          <td>2.845166</td>
          <td>2.773506</td>
          <td>2.533417</td>
          <td>-0.616746</td>
          <td>0.903326</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.562885</td>
          <td>2.205270</td>
          <td>1.080121</td>
          <td>1.192257</td>
          <td>1.240595</td>
          <td>1.411295</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.940721</td>
          <td>2.974182</td>
          <td>2.140298</td>
          <td>1.886342</td>
          <td>0.451992</td>
          <td>1.770786</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    m = np.array([[0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                  [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
                  [0.5, 0.0,-0.5, 0.0, 0.0, 0.0, 0.0],
                  [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    dot = make_dot(m)

    # Save pdf
    dot.render('dag')

    # Save png
    dot.format = 'png'
    dot.render('dag')

    dot




.. image:: ../image/bottom_up_parce.svg



Causal Discovery
----------------

To run causal discovery, we create a ``BottomUpParceLiNGAM`` object and
call the ``fit`` method.

.. code:: ipython3

    model = lingam.BottomUpParceLiNGAM()
    model.fit(X)




.. parsed-literal::

    <lingam.bottom_up_parce_lingam.BottomUpParceLiNGAM at 0x2652e878748>



Using the ``causal_order_`` properties, we can see the causal ordering
as a result of the causal discovery. x2 and x3, which have latent
confounders as parents, are stored in a list without causal ordering.

.. code:: ipython3

    model.causal_order_




.. parsed-literal::

    [[2, 3], 0, 5, 1, 4]



Also, using the ``adjacency_matrix_`` properties, we can see the
adjacency matrix as a result of the causal discovery. The coefficients
between variables with latent confounders are np.nan.

.. code:: ipython3

    model.adjacency_matrix_




.. parsed-literal::

    array([[ 0.   ,  0.   ,  0.   ,  0.506,  0.   ,  0.   ],
           [ 0.499,  0.   ,  0.495,  0.007,  0.   ,  0.   ],
           [ 0.   ,  0.   ,  0.   ,    nan,  0.   ,  0.   ],
           [ 0.   ,  0.   ,    nan,  0.   ,  0.   ,  0.   ],
           [ 0.448,  0.   , -0.451,  0.   ,  0.   ,  0.   ],
           [ 0.48 ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ]])



We can draw a causal graph by utility funciton.

.. code:: ipython3

    make_dot(model.adjacency_matrix_)




.. image:: ../image/bottom_up_parce2.svg



Bootstrapping
-------------

We call ``bootstrap()`` method instead of ``fit()``. Here, the second
argument specifies the number of bootstrap sampling.

.. code:: ipython3

    model = lingam.BottomUpParceLiNGAM()
    result = model.bootstrap(X, n_sampling=100)

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

    x4 <--- x0 (b>0) (58.0%)
    x4 <--- x2 (b<0) (58.0%)
    x1 <--- x0 (b>0) (52.0%)
    x1 <--- x2 (b>0) (51.0%)
    x5 <--- x0 (b>0) (32.0%)
    x1 <--- x3 (b>0) (28.0%)
    x0 <--- x3 (b>0) (18.0%)
    x5 <--- x2 (b>0) (10.0%)


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

    DAG[0]: 25.0%
    DAG[1]: 10.0%
    	x4 <--- x0 (b>0)
    	x4 <--- x2 (b<0)
    DAG[2]: 6.0%
    	x1 <--- x0 (b>0)
    	x1 <--- x2 (b>0)


Probability
-----------

Using the ``get_probabilities()`` method, we can get the probability of
bootstrapping.

.. code:: ipython3

    prob = result.get_probabilities(min_causal_effect=0.01)
    print(prob)


.. parsed-literal::

    [[0.   0.02 0.01 0.18 0.01 0.  ]
     [0.52 0.   0.51 0.28 0.   0.  ]
     [0.01 0.01 0.   0.06 0.   0.  ]
     [0.   0.   0.   0.   0.   0.  ]
     [0.58 0.04 0.58 0.02 0.   0.09]
     [0.32 0.03 0.1  0.02 0.   0.  ]]


Causal Effects
--------------

Using the ``get_causal_effects()`` method, we can get the list of causal
effect. The causal effects we can get are dictionary type variable. We
can display the list nicely by assigning it to pandas.DataFrame. Also,
we have replaced the variable index with a label below.

.. code:: ipython3

    causal_effects = result.get_causal_effects(min_causal_effect=0.01)

    # Assign to pandas.DataFrame for pretty display
    df = pd.DataFrame(causal_effects)
    labels = [f'x{i}' for i in range(X.shape[1])]
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
          <td>x4</td>
          <td>-0.126227</td>
          <td>0.58</td>
        </tr>
        <tr>
          <th>1</th>
          <td>x0</td>
          <td>x4</td>
          <td>0.060692</td>
          <td>0.58</td>
        </tr>
        <tr>
          <th>2</th>
          <td>x2</td>
          <td>x4</td>
          <td>-0.273587</td>
          <td>0.58</td>
        </tr>
        <tr>
          <th>3</th>
          <td>x0</td>
          <td>x1</td>
          <td>0.947100</td>
          <td>0.52</td>
        </tr>
        <tr>
          <th>4</th>
          <td>x3</td>
          <td>x1</td>
          <td>0.653876</td>
          <td>0.52</td>
        </tr>
        <tr>
          <th>5</th>
          <td>x2</td>
          <td>x1</td>
          <td>0.708409</td>
          <td>0.51</td>
        </tr>
        <tr>
          <th>6</th>
          <td>x1</td>
          <td>x4</td>
          <td>-0.050069</td>
          <td>0.42</td>
        </tr>
        <tr>
          <th>7</th>
          <td>x5</td>
          <td>x4</td>
          <td>0.029068</td>
          <td>0.41</td>
        </tr>
        <tr>
          <th>8</th>
          <td>x5</td>
          <td>x1</td>
          <td>0.758576</td>
          <td>0.38</td>
        </tr>
        <tr>
          <th>9</th>
          <td>x0</td>
          <td>x5</td>
          <td>0.517970</td>
          <td>0.32</td>
        </tr>
        <tr>
          <th>10</th>
          <td>x3</td>
          <td>x5</td>
          <td>0.265528</td>
          <td>0.32</td>
        </tr>
        <tr>
          <th>11</th>
          <td>x2</td>
          <td>x5</td>
          <td>0.225770</td>
          <td>0.30</td>
        </tr>
        <tr>
          <th>12</th>
          <td>x4</td>
          <td>x1</td>
          <td>0.044771</td>
          <td>0.30</td>
        </tr>
        <tr>
          <th>13</th>
          <td>x3</td>
          <td>x0</td>
          <td>0.511008</td>
          <td>0.18</td>
        </tr>
        <tr>
          <th>14</th>
          <td>x2</td>
          <td>x0</td>
          <td>0.415262</td>
          <td>0.16</td>
        </tr>
        <tr>
          <th>15</th>
          <td>x1</td>
          <td>x5</td>
          <td>-0.002557</td>
          <td>0.10</td>
        </tr>
        <tr>
          <th>16</th>
          <td>x3</td>
          <td>x2</td>
          <td>0.790837</td>
          <td>0.06</td>
        </tr>
        <tr>
          <th>17</th>
          <td>x4</td>
          <td>x5</td>
          <td>-0.004063</td>
          <td>0.06</td>
        </tr>
        <tr>
          <th>18</th>
          <td>x0</td>
          <td>x2</td>
          <td>0.421582</td>
          <td>0.04</td>
        </tr>
        <tr>
          <th>19</th>
          <td>x5</td>
          <td>x2</td>
          <td>0.393533</td>
          <td>0.04</td>
        </tr>
        <tr>
          <th>20</th>
          <td>x1</td>
          <td>x0</td>
          <td>0.569613</td>
          <td>0.02</td>
        </tr>
        <tr>
          <th>21</th>
          <td>x1</td>
          <td>x2</td>
          <td>0.596419</td>
          <td>0.01</td>
        </tr>
        <tr>
          <th>22</th>
          <td>x4</td>
          <td>x0</td>
          <td>0.081970</td>
          <td>0.01</td>
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
          <td>x0</td>
          <td>x1</td>
          <td>0.947100</td>
          <td>0.52</td>
        </tr>
        <tr>
          <th>16</th>
          <td>x3</td>
          <td>x2</td>
          <td>0.790837</td>
          <td>0.06</td>
        </tr>
        <tr>
          <th>8</th>
          <td>x5</td>
          <td>x1</td>
          <td>0.758576</td>
          <td>0.38</td>
        </tr>
        <tr>
          <th>5</th>
          <td>x2</td>
          <td>x1</td>
          <td>0.708409</td>
          <td>0.51</td>
        </tr>
        <tr>
          <th>4</th>
          <td>x3</td>
          <td>x1</td>
          <td>0.653876</td>
          <td>0.52</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    df.sort_values('probability', ascending=True).head()




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
          <th>22</th>
          <td>x4</td>
          <td>x0</td>
          <td>0.081970</td>
          <td>0.01</td>
        </tr>
        <tr>
          <th>21</th>
          <td>x1</td>
          <td>x2</td>
          <td>0.596419</td>
          <td>0.01</td>
        </tr>
        <tr>
          <th>20</th>
          <td>x1</td>
          <td>x0</td>
          <td>0.569613</td>
          <td>0.02</td>
        </tr>
        <tr>
          <th>19</th>
          <td>x5</td>
          <td>x2</td>
          <td>0.393533</td>
          <td>0.04</td>
        </tr>
        <tr>
          <th>18</th>
          <td>x0</td>
          <td>x2</td>
          <td>0.421582</td>
          <td>0.04</td>
        </tr>
      </tbody>
    </table>
    </div>



And with pandas.DataFrame, we can easily filter by keywords. The
following code extracts the causal direction towards x1.

.. code:: ipython3

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
          <th>3</th>
          <td>x0</td>
          <td>x1</td>
          <td>0.947100</td>
          <td>0.52</td>
        </tr>
        <tr>
          <th>4</th>
          <td>x3</td>
          <td>x1</td>
          <td>0.653876</td>
          <td>0.52</td>
        </tr>
        <tr>
          <th>5</th>
          <td>x2</td>
          <td>x1</td>
          <td>0.708409</td>
          <td>0.51</td>
        </tr>
        <tr>
          <th>8</th>
          <td>x5</td>
          <td>x1</td>
          <td>0.758576</td>
          <td>0.38</td>
        </tr>
        <tr>
          <th>12</th>
          <td>x4</td>
          <td>x1</td>
          <td>0.044771</td>
          <td>0.30</td>
        </tr>
      </tbody>
    </table>
    </div>



Because it holds the raw data of the causal effect (the original data
for calculating the median), it is possible to draw a histogram of the
values of the causal effect, as shown below.

.. code:: ipython3

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    %matplotlib inline

    from_index = 3 # index of x3
    to_index = 0 # index of x0
    plt.hist(result.total_effects_[:, to_index, from_index])




.. parsed-literal::

    (array([82.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 18.]),
     array([0.   , 0.051, 0.102, 0.153, 0.204, 0.256, 0.307, 0.358, 0.409,
            0.46 , 0.511]),
     <a list of 10 Patch objects>)




.. image:: ../image/bottom_up_parce_hist.png

