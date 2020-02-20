Tutorial
========
In this tutorial, we will show you how to run LiNGAM algorithms and see the results. We will also show you how to run the bootstrap method and check the results.

Requirements
------------

The following packages must be installed in order to run this tutorial. And import if necessary:

* numpy
* pandas
* scikit-learn
* graphviz

LiNGAM algorithm
----------------

First, we use lingam package:

.. code-block:: python

    import lingam

Then, if we want to run DirectLiNGAM algorithm, we create a :class:`~lingam.DirectLiNGAM` object and call the :func:`~lingam.DirectLiNGAM.fit` method:

.. code-block:: python

    model = lingam.DirectLiNGAM()
    model.fit(X)

* If you want to use the ICA-LiNGAM algorithm, replace :class:`~lingam.DirectLiNGAM` above with :class:`~lingam.ICALiNGAM`.

Using the :attr:`~lingam.DirectLiNGAM.causal_order_` property, we can see the causal ordering as a result of the causal discovery.

.. code-block:: python

    print(model.causal_order_)

The output of the :attr:`~lingam.DirectLiNGAM.causal_order_` property is as follows:

.. code-block:: python

    [3, 0, 2, 5, 1, 4]

Also, using the :attr:`~lingam.DirectLiNGAM.adjacency_matrix_` property, we can see the adjacency matrix as a result of the causal discovery.

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

For details, see also https://github.com/cdt15/lingam/blob/master/examples/DirectLiNGAM.ipynb

Bootstrapping
-------------

First, we create :class:`~lingam.DirectLiNGAM` object as above:

.. code-block:: python

    model = lingam.DirectLiNGAM()

Then, we call :func:`~lingam.DirectLiNGAM.bootstrap` method instead of :func:`~lingam.DirectLiNGAM.fit`. Here, the second argument specifies the number of bootstrap sampling.:

.. code-block:: python

    result = model.bootstrap(X, 100)

Causal Directions
^^^^^^^^^^^^^^^^^

Since :class:`~lingam.BootstrapResult` object is returned, we can get the ranking of the causal directions extracted by :func:`~lingam.BootstrapResult.get_causal_direction_counts` method. 

.. code-block:: python

    cdc = result.get_causal_direction_counts(n_directions=8, min_causal_effect=0.01)

For example, we can check the result as follows:

.. code-block:: python

    from lingam.utils import print_causal_directions
    print_causal_directions(cdc, 100)

.. code-block:: python

    x0 <--- x3 (100.0%)
    x1 <--- x0 (100.0%)
    x1 <--- x2 (100.0%)
    x2 <--- x3 (100.0%)
    x4 <--- x0 (100.0%)
    x4 <--- x2 (100.0%)
    x5 <--- x0 (100.0%)
    x0 <--- x2 (15.0%)

Increasing the value of `min_causal_effect` argument, causal directions with small effect does not output.

.. code-block:: python

    cdc = result.get_causal_direction_counts(n_directions=8, min_causal_effect=0.1)

.. code-block:: python

    print_causal_directions(cdc, 100)

.. code-block:: python

    x0 <--- x3 (100.0%)
    x1 <--- x0 (100.0%)
    x1 <--- x2 (100.0%)
    x2 <--- x3 (100.0%)
    x4 <--- x0 (100.0%)
    x4 <--- x2 (100.0%)
    x5 <--- x0 (100.0%)

By setting the value of `split_by_causal_effect_sign` argument to True, causal directions are split depending on the sign of the causal effect.

.. code-block:: python

    cdc = result.get_causal_direction_counts(n_directions=8, min_causal_effect=0.01, split_by_causal_effect_sign=True)

.. code-block:: python

    print_causal_directions(cdc, 100)

.. code-block:: python

    x0 <--- x3 (b>0) (100.0%)
    x1 <--- x0 (b>0) (100.0%)
    x1 <--- x2 (b>0) (100.0%)
    x2 <--- x3 (b>0) (100.0%)
    x4 <--- x0 (b>0) (100.0%)
    x4 <--- x2 (b<0) (100.0%)
    x5 <--- x0 (b>0) (100.0%)
    x0 <--- x2 (b>0) (8.0%)
    x0 <--- x2 (b<0) (7.0%)

Directed Acyclic Graphs
^^^^^^^^^^^^^^^^^^^^^^^

Also, using the :func:`~lingam.BootstrapResult.get_directed_acyclic_graph_counts` method, we can get the ranking of the DAGs extracted.

.. code-block:: python

    dagc = result.get_directed_acyclic_graph_counts(n_dags=3, min_causal_effect=0.01)

For example, we can check the result as follows:

.. code-block:: python

    from lingam.utils import print_dagc
    print_dagc(dagc, 100)

.. code-block:: python

    DAG[0]: 77.0%
        x0 <--- x3
        x1 <--- x0
        x1 <--- x2
        x2 <--- x3
        x4 <--- x0
        x4 <--- x2
    DAG[1]: 14.0%
        x0 <--- x2
        x0 <--- x3
        x1 <--- x0
        x1 <--- x2
        x2 <--- x3
        x4 <--- x0
        x4 <--- x2
    DAG[2]: 3.0%
        x0 <--- x3
        x1 <--- x0
        x1 <--- x2
        x2 <--- x3
        x4 <--- x0
        x4 <--- x2
        x4 <--- x5

Increasing the value of `min_causal_effect` argument, causal directions with small effect does not output.

.. code-block:: python

    dagc = result.get_directed_acyclic_graph_counts(n_dags=3, min_causal_effect=0.1)

.. code-block:: python

    print_dagc(dagc, 100)

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

By setting the value of `split_by_causal_effect_sign` argument to True, causal directions are split depending on the sign of the causal effect.

.. code-block:: python

    dagc = result.get_directed_acyclic_graph_counts(n_dags=3, min_causal_effect=0.01, split_by_causal_effect_sign=True)

.. code-block:: python

    print_dagc(dagc, 100)

.. code-block:: python

    DAG[0]: 77.0%
        x0 <--- x3 (b>0)
        x1 <--- x0 (b>0)
        x1 <--- x2 (b>0)
        x2 <--- x3 (b>0)
        x4 <--- x0 (b>0)
        x4 <--- x2 (b<0)
        x5 <--- x0 (b>0)
        x5 <--- x0 (b<0)
    DAG[1]: 14.0%
        x0 <--- x2 (b>0)
        x0 <--- x3 (b>0)
        x1 <--- x0 (b>0)
        x1 <--- x2 (b>0)
        x2 <--- x3 (b>0)
        x4 <--- x0 (b>0)
        x4 <--- x2 (b<0)
        x5 <--- x0 (b>0)
        x5 <--- x0 (b<0)
    DAG[2]: 3.0%
        x0 <--- x3 (b>0)
        x1 <--- x0 (b>0)
        x1 <--- x2 (b>0)
        x2 <--- x3 (b>0)
        x4 <--- x0 (b>0)
        x4 <--- x2 (b<0)
        x4 <--- x5 (b>0)
        x5 <--- x0 (b>0)
        x5 <--- x0 (b<0)

For details, see also https://github.com/cdt15/lingam/blob/master/examples/Bootstrap.ipynb


Use of Prior Knowledge
----------------------

we use lingam package and :func:`~lingam.utils.make_prior_knowledge`:

.. code-block:: python

    import lingam
    form lingam.utils import make_prior_knowledge

First, we create a prior knowledge matrix:

.. code-block:: python

    prior_knowledge = make_prior_knowledge(
        n_variables=6,
        exogenous_variables=[3],
        sink_variables=[1],
        paths=[[2, 4]],
        no_paths=[[4, 1], [1, 5]])
    print(prior_knowledge)

.. code-block:: python

    [[ 0  0 -1 -1 -1 -1]
     [-1  0 -1 -1  0 -1]
     [-1  0  0 -1  0 -1]
     [ 0  0  0  0  0  0]
     [-1  0  1 -1  0 -1]
     [-1  0 -1 -1 -1  0]]

Then, if we use a prior knowledge, we set prior knowledge matrix to :class:`~lingam.DirectLiNGAM` object:

.. code-block:: python

    model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
    model.fit(X)

Using the :attr:`~lingam.DirectLiNGAM.causal_order_` property, we can see the causal ordering as a result of the causal discovery.

.. code-block:: python

    print(model.causal_order_)

The output of the :attr:`~lingam.DirectLiNGAM.causal_order_` property is as follows:

.. code-block:: python

    [3, 2, 0, 4, 5, 1]

Also, using the :attr:`~lingam.DirectLiNGAM.adjacency_matrix_` property, we can see the adjacency matrix as a result of the causal discovery.

.. code-block:: python

    print(model.adjacency_matrix_)

The output of the :attr:`~lingam.DirectLiNGAM.adjacency_matrix_` property is as follows:

.. code-block:: python

    [[ 0.     0.     0.     2.992  0.     0.   ]
     [ 2.898  0.     1.965  0.     0.     0.   ]
     [ 0.     0.     0.     5.996  0.     0.   ]
     [ 0.     0.     0.     0.     0.     0.   ]
     [ 8.004  0.    -1.001  0.     0.     0.   ]
     [ 3.984  0.     0.     0.     0.     0.   ]]

For details, see also https://github.com/cdt15/lingam/blob/master/examples/DirectLiNGAM(PriorKnowledge).ipynb

Use of Multiple Dataset
-----------------------

We use lingam package:

.. code-block:: python

    import lingam

First, if we use two datasets, we create a list like this:

.. code-block:: python

    X_list = [X1, X2]

Then, we create a :class:`~lingam.MultiGroupDirectLiNGAM` object and call the :func:`~lingam.MultiGroupDirectLiNGAM.fit` method:

.. code-block:: python

    model = lingam.MultiGroupDirectLiNGAM()
    model.fit(X_list)

Using the :attr:`~lingam.MultiGroupDirectLiNGAM.causal_order_` property, we can see the causal ordering as a result of the causal discovery.

.. code-block:: python

    print(model.causal_order_)

Also, using the :attr:`~lingam.MultiGroupDirectLiNGAM.adjacency_matrices_` property, we can see the adjacency matrix as a result of the causal discovery. 
Since :attr:`~lingam.MultiGroupDirectLiNGAM.adjacency_matrices_` property returns a list, we can access the first matrix by indexing as follows:

.. code-block:: python

    print(model.adjacency_matrices_[0])

For details, see also https://github.com/cdt15/lingam/blob/master/examples/MultiGroupDirectLiNGAM.ipynb

Total Effect
------------

We use lingam package:

.. code-block:: python

    import lingam

Then, we create a :class:`~lingam.DirectLiNGAM` object and call the :func:`~lingam.DirectLiNGAM.fit` method:

.. code-block:: python

    model = lingam.DirectLiNGAM()
    model.fit(X)

To estimate the total effect, we can call :func:`~lingam.DirectLiNGAM.estimate_total_effect` method. The following example estimates the total effect from x3 to x1.

.. code-block:: python

    te = model.estimate_total_effect(X, 3, 1)
    print(f'total effect: {te:.3f}')

.. code-block:: python

    total effect: 21.002

For details, see also https://github.com/cdt15/lingam/blob/master/examples/TotalEffect.ipynb
