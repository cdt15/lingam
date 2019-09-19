Tutorial
========
In this tutorial, we will show you how to run LiNGAM algorithms and see the results. We will also show you how to run the bootstrap method and check the results.

Run LiNGAM algorithms
---------------------

First, we use lingam package:

.. code-block:: python

    import lingam

Then, if we want to run DirectLiNGAM algorithm, we create a :class:`~lingam.DirectLiNGAM` object and call the :func:`~lingam.DirectLiNGAM.fit` method:

.. code-block:: python

    model = lingam.DirectLiNGAM()
    model.fit(X)

Using the :attr:`~lingam.DirectLiNGAM.causal_order_` properties, we can see the causal ordering as a result of the causal discovery.

.. code-block:: python

    print(model.causal_order_)

The output of the :attr:`~lingam.DirectLiNGAM.causal_order_` property is as follows:

.. code-block:: python

    [3, 0, 2, 5, 1, 4]

Also, using the :attr:`~lingam.DirectLiNGAM.adjacency_matrix_` properties, we can see the adjacency matrix as a result of the causal discovery.

.. code-block:: python

    print(model.adjacency_matrix_)

The output of the :attr:`~lingam.DirectLiNGAM.adjacency_matrix_` property is as follows:

.. code-block:: python

    [[ 0.     0.     0.     3.006  0.     0.   ]
     [ 3.002  0.     1.996  0.     0.     0.   ]
     [ 0.     0.     0.     6.001  0.     0.   ]
     [ 0.     0.     0.     0.     0.     0.   ]
     [ 7.978  0.    -0.988  0.     0.     0.   ]
     [ 3.998  0.     0.     0.     0.     0.   ]]

For example, we can draw a causal graph by using graphviz as follows:

.. image:: image/dag.png

For details, see also https://github.com/cdt15/lingam/example/DirectLiNGAM.ipynb

Run Bootstrapping
-----------------

First, we create :class:`~lingam.DirectLiNGAM` object as above:

.. code-block:: python

    model = lingam.DirectLiNGAM()

Then, we call :func:`~lingam.DirectLiNGAM.bootstrap` method instead of :func:`~lingam.DirectLiNGAM.fit`. Here, the second argument specifies the number of bootstrap sampling.:

.. code-block:: python

    result = model.bootstrap(X, 100)

Since :class:`~lingam.BootstrapResult` object is returned, we can get the ranking of the causal directions extracted by :func:`~lingam.BootstrapResult.get_causal_direction_counts` method. 

.. code-block:: python

    cdc = result.get_causal_direction_counts(n_directions=8, min_causal_effect=0.01)

For example, we can check the result as follows:

.. code-block:: python

    for from_, to, count in zip(cdc['from'], cdc['to'], cdc['count']):
        print(f'x{to}' + ' <--- ' + f'x{from_}' + f' ({100*count/100:.1f}%)')

.. code-block:: python

    x0 <--- x3 (100.0%)
    x1 <--- x0 (100.0%)
    x1 <--- x2 (100.0%)
    x2 <--- x3 (100.0%)
    x4 <--- x0 (100.0%)
    x4 <--- x2 (100.0%)
    x5 <--- x0 (100.0%)
    x0 <--- x2 (15.0%)

Also, using the :func:`~lingam.BootstrapResult.get_directed_acyclic_graph_counts` method, we can get the ranking of the DAGs extracted.

.. code-block:: python

    dagc = result.get_directed_acyclic_graph_counts(n_dags=3, min_causal_effect=0.01)

For example, we can check the result as follows:

.. code-block:: python

    for i, (dag, count) in enumerate(zip(dagc['dag'], dagc['count'])):
        print(f'DAG[{i}]: {100*count/100:.1f}%')
        for from_, to in zip(dag['from'], dag['to']):
            print('\t' + f'x{to}' + ' <--- ' + f'x{from_}')

.. code-block:: python

    DAG[0]: 77.0%
        x0 <--- x3
        x1 <--- x0
        x1 <--- x2
        x2 <--- x3
        x4 <--- x0
        x4 <--- x2
        x5 <--- x0
    DAG[1]: 14.0%
        x0 <--- x2
        x0 <--- x3
        x1 <--- x0
        x1 <--- x2
        x2 <--- x3
        x4 <--- x0
        x4 <--- x2
        x5 <--- x0
    DAG[2]: 3.0%
        x0 <--- x3
        x1 <--- x0
        x1 <--- x2
        x2 <--- x3
        x4 <--- x0
        x4 <--- x2
        x4 <--- x5
        x5 <--- x0

For details, see also https://github.com/cdt15/lingam/example/Bootstrap.ipynb
