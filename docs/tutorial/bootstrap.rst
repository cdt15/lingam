
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
    
    import warnings
    warnings.filterwarnings("ignore")
    
    print([np.__version__, pd.__version__, graphviz.__version__, lingam.__version__])
    
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(0)


.. parsed-literal::

    ['1.26.4', '2.3.3', '0.21', '1.12.1']
    

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

    n_samples = 100
    
    model = lingam.DirectLiNGAM()
    result = model.bootstrap(X, n_sampling=n_samples)

Causal Directions
-----------------

Since :class:`~lingam.BootstrapResult` object is returned, we can get the ranking of the causal directions extracted by :func:`~lingam.BootstrapResult.get_causal_direction_counts` method. In the following sample code, ``n_directions`` option is limited to the causal directions of the top 8 rankings, and ``min_causal_effect`` option is limited to causal directions with a coefficient of 0.01 or more.

.. code-block:: python

    cdc = result.get_causal_direction_counts(n_directions=8, min_causal_effect=0.01, split_by_causal_effect_sign=True)

We can check the result by utility function.

.. code-block:: python

    print_causal_directions(cdc, n_samples)


.. parsed-literal::

    x5 <--- x0 (b>0) (100.0%)
    x1 <--- x0 (b>0) (100.0%)
    x1 <--- x2 (b>0) (100.0%)
    x4 <--- x2 (b<0) (100.0%)
    x0 <--- x3 (b>0) (98.0%)
    x4 <--- x0 (b>0) (98.0%)
    x2 <--- x3 (b>0) (96.0%)
    x3 <--- x2 (b>0) (4.0%)


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

    print_dagc(dagc, n_samples)


.. parsed-literal::

    DAG[0]: 84.0%
    	x0 <--- x3 (b>0)
    	x1 <--- x0 (b>0)
    	x1 <--- x2 (b>0)
    	x2 <--- x3 (b>0)
    	x4 <--- x0 (b>0)
    	x4 <--- x2 (b<0)
    	x5 <--- x0 (b>0)
    DAG[1]: 3.0%
    	x0 <--- x3 (b>0)
    	x1 <--- x0 (b>0)
    	x1 <--- x2 (b>0)
    	x3 <--- x2 (b>0)
    	x4 <--- x0 (b>0)
    	x4 <--- x2 (b<0)
    	x5 <--- x0 (b>0)
    DAG[2]: 2.0%
    	x0 <--- x3 (b>0)
    	x1 <--- x0 (b>0)
    	x1 <--- x2 (b>0)
    	x1 <--- x3 (b<0)
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

    [[0.   0.   0.03 0.98 0.02 0.  ]
     [1.   0.   1.   0.02 0.   0.01]
     [0.01 0.   0.   0.96 0.   0.01]
     [0.   0.   0.04 0.   0.   0.  ]
     [0.98 0.01 1.   0.02 0.   0.02]
     [1.   0.   0.02 0.02 0.   0.  ]]


Total Causal Effects
--------------------

Using the ``get_total_causal_effects()`` method, we can get the list of
total causal effect. The total causal effects we can get are dictionary
type variable. We can display the list nicely by assigning it to
pandas.DataFrame. Also, we have replaced the variable index with a label
below.

.. code-block:: python

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
          <td>3.004106</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>1</th>
          <td>x0</td>
          <td>x1</td>
          <td>2.963177</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>2</th>
          <td>x2</td>
          <td>x1</td>
          <td>2.017539</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>3</th>
          <td>x3</td>
          <td>x1</td>
          <td>20.928254</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>4</th>
          <td>x0</td>
          <td>x5</td>
          <td>3.997787</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>5</th>
          <td>x3</td>
          <td>x4</td>
          <td>18.077943</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>6</th>
          <td>x3</td>
          <td>x5</td>
          <td>12.012988</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>7</th>
          <td>x2</td>
          <td>x4</td>
          <td>-1.006362</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>8</th>
          <td>x0</td>
          <td>x4</td>
          <td>8.011818</td>
          <td>0.98</td>
        </tr>
        <tr>
          <th>9</th>
          <td>x3</td>
          <td>x2</td>
          <td>5.964879</td>
          <td>0.96</td>
        </tr>
        <tr>
          <th>10</th>
          <td>x2</td>
          <td>x5</td>
          <td>0.396327</td>
          <td>0.09</td>
        </tr>
        <tr>
          <th>11</th>
          <td>x2</td>
          <td>x0</td>
          <td>0.487915</td>
          <td>0.07</td>
        </tr>
        <tr>
          <th>12</th>
          <td>x2</td>
          <td>x3</td>
          <td>0.164565</td>
          <td>0.04</td>
        </tr>
        <tr>
          <th>13</th>
          <td>x5</td>
          <td>x4</td>
          <td>0.087437</td>
          <td>0.03</td>
        </tr>
        <tr>
          <th>14</th>
          <td>x4</td>
          <td>x5</td>
          <td>0.496445</td>
          <td>0.02</td>
        </tr>
        <tr>
          <th>15</th>
          <td>x5</td>
          <td>x1</td>
          <td>-0.064703</td>
          <td>0.02</td>
        </tr>
        <tr>
          <th>16</th>
          <td>x4</td>
          <td>x1</td>
          <td>0.367100</td>
          <td>0.02</td>
        </tr>
        <tr>
          <th>17</th>
          <td>x4</td>
          <td>x0</td>
          <td>0.124114</td>
          <td>0.02</td>
        </tr>
        <tr>
          <th>18</th>
          <td>x0</td>
          <td>x2</td>
          <td>0.056261</td>
          <td>0.01</td>
        </tr>
        <tr>
          <th>19</th>
          <td>x1</td>
          <td>x4</td>
          <td>-0.097108</td>
          <td>0.01</td>
        </tr>
        <tr>
          <th>20</th>
          <td>x5</td>
          <td>x2</td>
          <td>-0.111894</td>
          <td>0.01</td>
        </tr>
      </tbody>
    </table>
    </div>



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
          <td>20.928254</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>5</th>
          <td>x3</td>
          <td>x4</td>
          <td>18.077943</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>6</th>
          <td>x3</td>
          <td>x5</td>
          <td>12.012988</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>8</th>
          <td>x0</td>
          <td>x4</td>
          <td>8.011818</td>
          <td>0.98</td>
        </tr>
        <tr>
          <th>9</th>
          <td>x3</td>
          <td>x2</td>
          <td>5.964879</td>
          <td>0.96</td>
        </tr>
      </tbody>
    </table>
    </div>



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
          <th>20</th>
          <td>x5</td>
          <td>x2</td>
          <td>-0.111894</td>
          <td>0.01</td>
        </tr>
        <tr>
          <th>18</th>
          <td>x0</td>
          <td>x2</td>
          <td>0.056261</td>
          <td>0.01</td>
        </tr>
        <tr>
          <th>19</th>
          <td>x1</td>
          <td>x4</td>
          <td>-0.097108</td>
          <td>0.01</td>
        </tr>
        <tr>
          <th>17</th>
          <td>x4</td>
          <td>x0</td>
          <td>0.124114</td>
          <td>0.02</td>
        </tr>
        <tr>
          <th>16</th>
          <td>x4</td>
          <td>x1</td>
          <td>0.367100</td>
          <td>0.02</td>
        </tr>
      </tbody>
    </table>
    </div>



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
          <td>2.963177</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>2</th>
          <td>x2</td>
          <td>x1</td>
          <td>2.017539</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>3</th>
          <td>x3</td>
          <td>x1</td>
          <td>20.928254</td>
          <td>1.00</td>
        </tr>
        <tr>
          <th>15</th>
          <td>x5</td>
          <td>x1</td>
          <td>-0.064703</td>
          <td>0.02</td>
        </tr>
        <tr>
          <th>16</th>
          <td>x4</td>
          <td>x1</td>
          <td>0.367100</td>
          <td>0.02</td>
        </tr>
      </tbody>
    </table>
    </div>


Because it holds the raw data of the total causal effect (the original data
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

Furthermore, when we separate the bootstrap coefficient distributions
into the three structural cases - X->Y, Y->X, and no directed edge between
X and Y - the resulting histograms are shown below.

.. code-block:: python

    import matplotlib.ticker as ticker
    
    from_index, to_index = 0, 4
    
    te_xy = result.total_effects_[:, to_index, from_index]
    te_yx = result.total_effects_[:, from_index, to_index]
    
    both_zero_mask = (te_xy == 0.0) & (te_yx == 0.0)
    te_zero = result.total_effects_[both_zero_mask, to_index, from_index]
    
    te_xy = te_xy[te_xy != 0.0]
    te_yx = te_yx[te_yx != 0.0]
    
    bins_count = int(np.ceil(1 + np.log2(max(n_samples, 1))))
    
    # calculate xmin, xmax
    arr_list = [te_xy, te_yx, te_zero]
    if any(a.size > 0 for a in arr_list):
        vals = np.concatenate([a for a in arr_list if a.size > 0])
    else:
        vals = np.array([0.0])
    
    xmin, xmax = np.min(vals), np.max(vals)
    if xmin == xmax:
        eps = 1e-9 if xmin == 0 else abs(xmin) * 1e-3
        xmin, xmax = xmin - eps, xmax + eps
    
    bin_edges = np.linspace(xmin, xmax, bins_count + 1)
    
    # calculate ymax
    counts_xy, _ = np.histogram(te_xy, bins=bin_edges) if te_xy.size > 0 else (np.zeros(bins_count, dtype=int), None)
    counts_yx, _ = np.histogram(te_yx, bins=bin_edges) if te_yx.size > 0 else (np.zeros(bins_count, dtype=int), None)
    counts_zz, _ = np.histogram(te_zero, bins=bin_edges) if te_zero.size > 0 else (np.zeros(bins_count, dtype=int), None)
    
    ymax = int(max(counts_xy.max(initial=0), counts_yx.max(initial=0), counts_zz.max(initial=0)))
    ymax = max(ymax, 1)
    # If you want to set ymax to the number of bootstrap iterations, uncomment next line.
    # ymax = n_samples
    
    # display histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    labels = [f'x{i}' for i in range(X.shape[1])]
    
    axes[0].hist(te_xy, bins=bin_edges)
    axes[0].set_title(f"{labels[from_index]} -> {labels[to_index]}")
    axes[0].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axes[0].set_xlim(xmin, xmax)
    axes[0].set_ylim(0, ymax)
    
    axes[1].hist(te_yx, bins=bin_edges)
    axes[1].set_title(f"{labels[to_index]} -> {labels[from_index]}")
    axes[1].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axes[1].set_xlim(xmin, xmax)
    axes[1].set_ylim(0, ymax)
    
    axes[2].hist(te_zero, bins=bin_edges)
    axes[2].set_title("No directed edge between " + labels[from_index] + " and " + labels[to_index])
    axes[2].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axes[2].set_xlim(xmin, xmax)
    axes[2].set_ylim(0, ymax)
    
    plt.tight_layout()
    plt.show()



.. image:: ../image/bootstrap_hists.png

Bootstrap Probability of Path
-----------------------------

Using the ``get_paths()`` method, we can explore all paths from any
variable to any variable and calculate the bootstrap probability for
each path. The path will be output as an array of variable indices. For
example, the array ``[3, 0, 1]`` shows the path from variable X3 through
variable X0 to variable X1.

.. code-block:: python

    from_index = 3 # index of x3
    to_index = 1 # index of x1
    
    pd.DataFrame(result.get_paths(from_index, to_index))




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
          <th>path</th>
          <th>effect</th>
          <th>probability</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>[3, 0, 1]</td>
          <td>8.893562</td>
          <td>0.98</td>
        </tr>
        <tr>
          <th>1</th>
          <td>[3, 2, 1]</td>
          <td>12.030408</td>
          <td>0.96</td>
        </tr>
        <tr>
          <th>2</th>
          <td>[3, 2, 0, 1]</td>
          <td>2.239175</td>
          <td>0.03</td>
        </tr>
        <tr>
          <th>3</th>
          <td>[3, 1]</td>
          <td>-0.639462</td>
          <td>0.02</td>
        </tr>
        <tr>
          <th>4</th>
          <td>[3, 2, 4, 0, 1]</td>
          <td>-3.194541</td>
          <td>0.02</td>
        </tr>
        <tr>
          <th>5</th>
          <td>[3, 4, 0, 1]</td>
          <td>9.820705</td>
          <td>0.02</td>
        </tr>
        <tr>
          <th>6</th>
          <td>[3, 0, 2, 1]</td>
          <td>3.061033</td>
          <td>0.01</td>
        </tr>
        <tr>
          <th>7</th>
          <td>[3, 0, 5, 1]</td>
          <td>1.176834</td>
          <td>0.01</td>
        </tr>
        <tr>
          <th>8</th>
          <td>[3, 0, 5, 2, 1]</td>
          <td>-2.719517</td>
          <td>0.01</td>
        </tr>
      </tbody>
    </table>
    </div>


