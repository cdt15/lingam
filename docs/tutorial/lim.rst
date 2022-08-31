
LiM
========

Import and settings
-------------------

In this example, we need to import ``numpy``, and ``random``, 
in addition to ``lingam``.

.. code-block:: python

    import numpy as np
    import random
    import lingam
    import lingam.utils as ut
        
    print([np.__version__, lingam.__version__])


.. parsed-literal::

    ['1.20.3', '1.6.0']
    

Test data 
-----------

First, we generate a causal structure with 2 variables, where one of them is randomly set to be a discrete variable.

.. code-block:: python

    ut.set_random_seed(1)
    n_samples, n_features, n_edges, graph_type, sem_type = 1000, 2, 1, 'ER', 'mixed_random_i_dis'
    B_true = ut.simulate_dag(n_features, n_edges, graph_type)
    W_true = ut.simulate_parameter(B_true)  # row to column

    no_dis = np.random.randint(1, n_features)  # number of discrete vars.
    print('There are %d discrete variable(s).' % (no_dis))
    nodes = [iii for iii in range(n_features)]
    dis_var = random.sample(nodes, no_dis) # randomly select no_dis discrete variables
    dis_con = np.full((1, n_features), np.inf)
    for iii in range(n_features):
        if iii in dis_var:
            dis_con[0, iii] = 0  # 1:continuous;   0:discrete
        else:
            dis_con[0, iii] = 1
    
    X = ut.simulate_linear_mixed_sem(W_true, n_samples, sem_type, dis_con)

    print('The true adjacency matrix is:\n', W_true)

   
.. parsed-literal::

    There are 1 discrete variable(s).
    The true adjacency matrix is:
    [[0.        0.       ]
    [1.3082251 0.       ]]


Causal Discovery for linear mixed data
-----------------------------------------

To run causal discovery, we create a ``LiM`` object and call the ``fit``
method.

.. code-block:: python

    model = lingam.LiM()
    model.fit(X, dis_con)



.. parsed-literal::

    <lingam.lim.LiM at 0x174d475f850>



Using the ``_adjacency_matrix`` properties, we can see the estimated adjacency matrix between mixed variables.



.. code-block:: python

    print('The estimated adjacency matrix is:\n', model._adjacency_matrix)



.. parsed-literal::

    The estimated adjacency matrix is:
    [[ 0.        ,  0.        ],
     [-1.09938457,  0.        ]]
