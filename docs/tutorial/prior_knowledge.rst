Use of Prior Knowledge
======================

We use lingam package and :func:`~lingam.utils.make_prior_knowledge`:

.. code-block:: python

    import lingam
    from lingam.utils import make_prior_knowledge

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

The values of the prior knowledge matrix elements are represented as follows:

* ``0`` : :math:`x_i` does not have a directed path to :math:`x_j`
* ``1`` : :math:`x_i` has a directed path to :math:`x_j`
* ``-1`` : No prior knowledge is available to know if either of the two cases above (0 or 1) is true.

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
