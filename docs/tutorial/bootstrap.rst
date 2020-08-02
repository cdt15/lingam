
Bootstrap
=========

Import and settings
-------------------

In this example, we need to import ``numpy``, ``pandas``, and ``graphviz`` in addition to ``lingam``.

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

    ['1.16.2', '0.24.2', '0.11.1', '1.3.1']
    

Test data
---------

We create test data consisting of 6 variables.

.. code-block:: python

    x3 = np.random.uniform(size=1000)
    x0 = 3.0*x3 + np.random.uniform(size=1000)
    x2 = 6.0*x3 + np.random.uniform(size=1000)
    x1 = 3.0*x0 + 2.0*x2 + np.random.uniform(size=1000)
    x5 = 4.0*x0 + np.random.uniform(size=1000)
    x4 = 8.0*x0 - 1.0*x2 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T ,columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])
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
    <br>



.. code-block:: python

    m = np.array([[0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                  [3.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [8.0, 0.0,-1.0, 0.0, 0.0, 0.0],
                  [4.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    
    make_dot(m)




.. image:: ../image/bootstrap_dag.svg



Bootstrapping
-------------

We call :func:`~lingam.DirectLiNGAM.bootstrap` method instead of :func:`~lingam.DirectLiNGAM.fit`. Here, the second argument specifies the number of bootstrap sampling.

.. code-block:: python

    model = lingam.DirectLiNGAM()
    result = model.bootstrap(X, n_sampling=100)

Causal Directions
-----------------

Since :class:`~lingam.BootstrapResult` object is returned, we can get the ranking of the causal directions extracted by :func:`~lingam.BootstrapResult.get_causal_direction_counts` method. In the following sample code, ``n_directions`` option is limited to the causal directions of the top 8 rankings, and ``min_causal_effect`` option is limited to causal directions with a coefficient of 0.01 or more.

.. code-block:: python

    cdc = result.get_causal_direction_counts(n_directions=8, min_causal_effect=0.01, split_by_causal_effect_sign=True)

We can check the result by utility function.

.. code-block:: python

    print_causal_directions(cdc, 100)


.. parsed-literal::

    x1 <--- x0 (b>0) (100.0%)
    x1 <--- x2 (b>0) (100.0%)
    x5 <--- x0 (b>0) (100.0%)
    x0 <--- x3 (b>0) (99.0%)
    x4 <--- x0 (b>0) (98.0%)
    x2 <--- x3 (b>0) (96.0%)
    x4 <--- x2 (b<0) (94.0%)
    x4 <--- x5 (b>0) (20.0%)
    

Directed Acyclic Graphs
-----------------------

Also, using the :func:`~lingam.BootstrapResult.get_directed_acyclic_graph_counts()` method, we can
get the ranking of the DAGs extracted. In the following sample code,
``n_dags`` option is limited to the dags of the top 3 rankings, and
``min_causal_effect`` option is limited to causal directions with a
coefficient of 0.01 or more.

.. code-block:: python

    dagc = result.get_directed_acyclic_graph_counts(n_dags=3, min_causal_effect=0.01, split_by_causal_effect_sign=True)

We can check the result by utility function.

.. code-block:: python

    print_dagc(dagc, 100)


.. parsed-literal::

    DAG[0]: 54.0%
    	x0 <--- x3 (b>0)
    	x1 <--- x0 (b>0)
    	x1 <--- x2 (b>0)
    	x2 <--- x3 (b>0)
    	x4 <--- x0 (b>0)
    	x4 <--- x2 (b<0)
    	x5 <--- x0 (b>0)
    DAG[1]: 16.0%
    	x0 <--- x3 (b>0)
    	x1 <--- x0 (b>0)
    	x1 <--- x2 (b>0)
    	x2 <--- x3 (b>0)
    	x4 <--- x0 (b>0)
    	x4 <--- x2 (b<0)
    	x4 <--- x5 (b>0)
    	x5 <--- x0 (b>0)
    DAG[2]: 7.0%
    	x0 <--- x3 (b>0)
    	x1 <--- x0 (b>0)
    	x1 <--- x2 (b>0)
    	x1 <--- x3 (b>0)
    	x2 <--- x3 (b>0)
    	x4 <--- x0 (b>0)
    	x4 <--- x2 (b<0)
    	x5 <--- x0 (b>0)
    

Probability
-----------

Using the :func:`~lingam.BootstrapResult.get_probabilities()` method, we can get the probability of
bootstrapping.

.. code-block:: python

    prob = result.get_probabilities(min_causal_effect=0.01)
    print(prob)


.. parsed-literal::

    [[0.   0.   0.1  0.99 0.02 0.  ]
     [1.   0.   1.   0.1  0.   0.05]
     [0.   0.   0.   0.96 0.   0.  ]
     [0.   0.   0.04 0.   0.   0.  ]
     [0.98 0.   0.94 0.02 0.   0.2 ]
     [1.   0.   0.   0.   0.   0.  ]]
    

Causal Effects
--------------

Using the :func:`~lingam.BootstrapResult.get_causal_effects()` method, we can get the list of causal
effect. The causal effects we can get are dictionary type variable. We
can display the list nicely by assigning it to pandas.DataFrame. Also,
we have replaced the variable index with a label below.

.. code-block:: python

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
          <td>x0</td>
          <td>3.006190</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>1</th>
          <td>x0</td>
          <td>x1</td>
          <td>3.004868</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>2</th>
          <td>x2</td>
          <td>x1</td>
          <td>2.092102</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>3</th>
          <td>x3</td>
          <td>x1</td>
          <td>20.931938</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>4</th>
          <td>x0</td>
          <td>x5</td>
          <td>3.982892</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>5</th>
          <td>x3</td>
          <td>x5</td>
          <td>12.024250</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>6</th>
          <td>x2</td>
          <td>x4</td>
          <td>-0.887620</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>7</th>
          <td>x3</td>
          <td>x4</td>
          <td>18.077244</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>8</th>
          <td>x0</td>
          <td>x4</td>
          <td>7.993145</td>
          <td>0.98</td>
        </tr>
        <tr>
          <th>9</th>
          <td>x3</td>
          <td>x2</td>
          <td>5.970163</td>
          <td>0.96</td>
        </tr>
        <tr>
          <th>10</th>
          <td>x5</td>
          <td>x1</td>
          <td>0.011708</td>
          <td>0.79</td>
        </tr>
        <tr>
          <th>11</th>
          <td>x2</td>
          <td>x5</td>
          <td>0.024284</td>
          <td>0.72</td>
        </tr>
        <tr>
          <th>12</th>
          <td>x0</td>
          <td>x2</td>
          <td>0.014228</td>
          <td>0.70</td>
        </tr>
        <tr>
          <th>13</th>
          <td>x5</td>
          <td>x4</td>
          <td>0.015170</td>
          <td>0.66</td>
        </tr>
        <tr>
          <th>14</th>
          <td>x2</td>
          <td>x0</td>
          <td>0.015480</td>
          <td>0.30</td>
        </tr>
        <tr>
          <th>15</th>
          <td>x1</td>
          <td>x5</td>
          <td>0.021215</td>
          <td>0.21</td>
        </tr>
        <tr>
          <th>16</th>
          <td>x4</td>
          <td>x1</td>
          <td>-0.004251</td>
          <td>0.04</td>
        </tr>
        <tr>
          <th>17</th>
          <td>x2</td>
          <td>x3</td>
          <td>0.163050</td>
          <td>0.04</td>
        </tr>
        <tr>
          <th>18</th>
          <td>x4</td>
          <td>x0</td>
          <td>0.122301</td>
          <td>0.02</td>
        </tr>
        <tr>
          <th>19</th>
          <td>x4</td>
          <td>x5</td>
          <td>0.009574</td>
          <td>0.02</td>
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
          <td>20.931938</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>7</th>
          <td>x3</td>
          <td>x4</td>
          <td>18.077244</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>5</th>
          <td>x3</td>
          <td>x5</td>
          <td>12.024250</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>8</th>
          <td>x0</td>
          <td>x4</td>
          <td>7.993145</td>
          <td>0.98</td>
        </tr>
        <tr>
          <th>9</th>
          <td>x3</td>
          <td>x2</td>
          <td>5.970163</td>
          <td>0.96</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br>



.. code-block:: python

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
          <th>19</th>
          <td>x4</td>
          <td>x5</td>
          <td>0.009574</td>
          <td>0.02</td>
        </tr>
        <tr>
          <th>18</th>
          <td>x4</td>
          <td>x0</td>
          <td>0.122301</td>
          <td>0.02</td>
        </tr>
        <tr>
          <th>17</th>
          <td>x2</td>
          <td>x3</td>
          <td>0.163050</td>
          <td>0.04</td>
        </tr>
        <tr>
          <th>16</th>
          <td>x4</td>
          <td>x1</td>
          <td>-0.004251</td>
          <td>0.04</td>
        </tr>
        <tr>
          <th>15</th>
          <td>x1</td>
          <td>x5</td>
          <td>0.021215</td>
          <td>0.21</td>
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
          <td>3.004868</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>2</th>
          <td>x2</td>
          <td>x1</td>
          <td>2.092102</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>3</th>
          <td>x3</td>
          <td>x1</td>
          <td>20.931938</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>10</th>
          <td>x5</td>
          <td>x1</td>
          <td>0.011708</td>
          <td>0.79</td>
        </tr>
        <tr>
          <th>16</th>
          <td>x4</td>
          <td>x1</td>
          <td>-0.004251</td>
          <td>0.04</td>
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
    
    from_index = 3 # index of x3
    to_index = 0 # index of x0
    plt.hist(result.total_effects_[:, to_index, from_index])


.. image:: ../image/bootstrap_hist.png

