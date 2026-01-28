EvaluateModelFit
================

This notebook explains how to use ``lingam.utils.evaluate_model_fit``.
This function returns the mode fit of the given adjacency matrix to the
data.

Import and settings
-------------------

.. code-block:: python

    import numpy as np
    import pandas as pd
    from scipy.special import expit
    import lingam
    from lingam.utils import make_dot
    
    print([np.__version__, pd.__version__, lingam.__version__])
    
    import warnings
    warnings.filterwarnings("ignore")
    
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(100)


.. parsed-literal::

    ['1.26.4', '2.3.3', '1.12.1']
    

When all variables are continuous data
--------------------------------------

Test data
~~~~~~~~~

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
          <td>1.657947</td>
          <td>12.090323</td>
          <td>3.519873</td>
          <td>0.543405</td>
          <td>10.182785</td>
          <td>7.401408</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.217345</td>
          <td>7.607388</td>
          <td>1.693219</td>
          <td>0.278369</td>
          <td>8.758949</td>
          <td>4.912979</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2.226804</td>
          <td>13.483555</td>
          <td>3.201513</td>
          <td>0.424518</td>
          <td>15.398626</td>
          <td>9.098729</td>
        </tr>
        <tr>
          <th>3</th>
          <td>2.756527</td>
          <td>20.654225</td>
          <td>6.037873</td>
          <td>0.844776</td>
          <td>16.795156</td>
          <td>11.147294</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.319283</td>
          <td>3.340782</td>
          <td>0.727265</td>
          <td>0.004719</td>
          <td>2.343100</td>
          <td>2.037974</td>
        </tr>
      </tbody>
    </table>
    </div>



Causal Discovery
~~~~~~~~~~~~~~~~

Perform causal discovery to obtain the adjacency matrix.

.. code-block:: python

    model = lingam.DirectLiNGAM()
    model.fit(X)
    model.adjacency_matrix_




.. parsed-literal::

    array([[ 0.   ,  0.   ,  0.   ,  2.994,  0.   ,  0.   ],
           [ 2.995,  0.   ,  1.993,  0.   ,  0.   ,  0.   ],
           [ 0.   ,  0.   ,  0.   ,  5.957,  0.   ,  0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
           [ 7.998,  0.   , -1.005,  0.   ,  0.   ,  0.   ],
           [ 3.98 ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ]])



Evaluation
~~~~~~~~~~

Calculate the model fit of the given adjacency matrix to given data.

.. code-block:: python

    lingam.utils.evaluate_model_fit(model.adjacency_matrix_, X)




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
          <th>DoF</th>
          <th>DoF Baseline</th>
          <th>chi2</th>
          <th>chi2 p-value</th>
          <th>chi2 Baseline</th>
          <th>CFI</th>
          <th>GFI</th>
          <th>AGFI</th>
          <th>NFI</th>
          <th>TLI</th>
          <th>RMSEA</th>
          <th>AIC</th>
          <th>BIC</th>
          <th>LogLik</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Value</th>
          <td>9</td>
          <td>16</td>
          <td>11.129623</td>
          <td>0.266928</td>
          <td>22997.243286</td>
          <td>0.999907</td>
          <td>0.999516</td>
          <td>0.99914</td>
          <td>0.999516</td>
          <td>0.999835</td>
          <td>0.01539</td>
          <td>23.977741</td>
          <td>82.870804</td>
          <td>0.01113</td>
        </tr>
      </tbody>
    </table>
    </div>



When the data has hidden common causes
--------------------------------------

Test data
~~~~~~~~~

.. code-block:: python

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
          <td>0.978424</td>
          <td>1.966955</td>
          <td>1.219048</td>
          <td>1.746943</td>
          <td>0.761499</td>
          <td>0.942972</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.164124</td>
          <td>2.652780</td>
          <td>2.153412</td>
          <td>2.317986</td>
          <td>0.427684</td>
          <td>1.144585</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.160532</td>
          <td>1.978590</td>
          <td>0.919055</td>
          <td>1.066110</td>
          <td>0.603656</td>
          <td>1.329139</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.502959</td>
          <td>1.833784</td>
          <td>1.748939</td>
          <td>1.234851</td>
          <td>0.447353</td>
          <td>1.188017</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.948636</td>
          <td>2.457468</td>
          <td>1.535006</td>
          <td>2.073317</td>
          <td>0.501208</td>
          <td>1.155161</td>
        </tr>
      </tbody>
    </table>
    </div>



Causal Discovery
~~~~~~~~~~~~~~~~

nan represents having a hidden common cause.

.. code-block:: python

    model = lingam.BottomUpParceLiNGAM()
    model.fit(X)
    model.adjacency_matrix_




.. parsed-literal::

    array([[ 0.   ,    nan,  0.   ,    nan,  0.   ,  0.   ],
           [   nan,  0.   ,  0.   ,    nan,  0.   ,  0.   ],
           [-0.22 ,  0.593,  0.   ,  0.564,  0.   ,  0.   ],
           [   nan,    nan,  0.   ,  0.   ,  0.   ,  0.   ],
           [ 0.542,  0.   , -0.529,  0.   ,  0.   ,  0.   ],
           [ 0.506,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ]])



.. code-block:: python

    lingam.utils.evaluate_model_fit(model.adjacency_matrix_, X)




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
          <th>DoF</th>
          <th>DoF Baseline</th>
          <th>chi2</th>
          <th>chi2 p-value</th>
          <th>chi2 Baseline</th>
          <th>CFI</th>
          <th>GFI</th>
          <th>AGFI</th>
          <th>NFI</th>
          <th>TLI</th>
          <th>RMSEA</th>
          <th>AIC</th>
          <th>BIC</th>
          <th>LogLik</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Value</th>
          <td>-3</td>
          <td>15</td>
          <td>1673.491733</td>
          <td>NaN</td>
          <td>4158.502617</td>
          <td>0.595393</td>
          <td>0.597573</td>
          <td>3.012133</td>
          <td>0.597573</td>
          <td>3.023037</td>
          <td>NaN</td>
          <td>44.653017</td>
          <td>162.439143</td>
          <td>1.673492</td>
        </tr>
      </tbody>
    </table>
    </div>



When the data has ordinal variables
-----------------------------------

Test data
~~~~~~~~~

.. code-block:: python

    x3 = np.random.uniform(size=1000)
    x0 = 0.6*x3 + np.random.uniform(size=1000)
    
    # discrete
    x2 = 1.2*x3 + np.random.uniform(size=1000)
    x2 = expit(x2 - np.mean(x2))
    vec_func = np.vectorize(lambda p: np.random.choice([0, 1], p=[p, 1 - p]))
    x2 = vec_func(x2)
    
    x1 = 0.6*x0 + 0.4*x2 + np.random.uniform(size=1000)
    x5 = 0.8*x0 + np.random.uniform(size=1000)
    x4 = 1.6*x0 - 0.2*x2 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T ,columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])
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
          <td>0.471823</td>
          <td>1.426239</td>
          <td>1.0</td>
          <td>0.129133</td>
          <td>1.535926</td>
          <td>0.567324</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.738933</td>
          <td>1.723219</td>
          <td>1.0</td>
          <td>0.327512</td>
          <td>1.806484</td>
          <td>1.056211</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.143877</td>
          <td>1.962664</td>
          <td>1.0</td>
          <td>0.538189</td>
          <td>2.075554</td>
          <td>1.865132</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.326486</td>
          <td>0.946426</td>
          <td>1.0</td>
          <td>0.302415</td>
          <td>0.675984</td>
          <td>0.857528</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.942822</td>
          <td>0.882616</td>
          <td>0.0</td>
          <td>0.529399</td>
          <td>2.002522</td>
          <td>1.063416</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code-block:: python

    adjacency_matrix = np.array([
        [0.0, 0.0, 0.0, 0.6, 0.0, 0.0],
        [0.6, 0.0, 0.4, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.2, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.6, 0.0,-0.2, 0.0, 0.0, 0.0],
        [0.8, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )

Specify whether each variable is an ordinal variable in ``is_ordinal``.

.. code-block:: python

    lingam.utils.evaluate_model_fit(adjacency_matrix, X, is_ordinal=[0, 0, 1, 0, 0, 0])




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
          <th>DoF</th>
          <th>DoF Baseline</th>
          <th>chi2</th>
          <th>chi2 p-value</th>
          <th>chi2 Baseline</th>
          <th>CFI</th>
          <th>GFI</th>
          <th>AGFI</th>
          <th>NFI</th>
          <th>TLI</th>
          <th>RMSEA</th>
          <th>AIC</th>
          <th>BIC</th>
          <th>LogLik</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Value</th>
          <td>9</td>
          <td>16</td>
          <td>19.949525</td>
          <td>0.018226</td>
          <td>2733.058196</td>
          <td>0.99597</td>
          <td>0.992701</td>
          <td>0.987023</td>
          <td>0.992701</td>
          <td>0.992836</td>
          <td>0.034897</td>
          <td>23.960101</td>
          <td>82.853164</td>
          <td>0.01995</td>
        </tr>
      </tbody>
    </table>
    </div>


