LiNGAM algorithm
================

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

.. image:: ../image/dag.png

For details, see also:

* https://github.com/cdt15/lingam/blob/master/examples/DirectLiNGAM.ipynb
* https://github.com/cdt15/lingam/blob/master/examples/DirectLiNGAM(Kernel).ipynb
