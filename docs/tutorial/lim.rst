
LiM
========

Model
-------------------
Linear Mixed (LiM) causal discovery algorithm [1]_ extends LiNGAM to handle the mixed data that consists of both continuous and discrete variables. 
The estimation is performed by first globally optimizing the log-likelihood function on the joint distribution of data with the acyclicity constraint, and then applying a local combinatorial search to output a causal graph.

This method is based on the LiM model as shown below, 

i) As for the continuous variable, its value assigned to each of :math:`x_i` is a linear function of its parent variables denoted 
by :math:`x_{\mathrm{pa}(i)}` plus a non-Gaussian error term :math:`e_i`, that is, 

.. math::
    
    x_i = e_i + c_i + \sum_{j \in \mathrm{pa}(i) }{b_{ij} x_j}, \quad e_i \sim Non-Gaussian(\cdot),

where the error terms :math:`e_i` are continuous random variables with non-Gaussian densities, and the error variables :math:`e_i` are independent of 
each other. The coefficients :math:`b_{ij}` and intercepts :math:`c_i` are constants.

ii) As for the discrete variable, its value equals 1 if the linear function of its parent variables :math:`x_{\mathrm{pa}(i)}` plus a Logistic error 
term :math:`e_i` is larger than 0, otherwise, its value equals 0. That is, 

.. math::
   :nowrap:

   \begin{eqnarray}
          x_i = \begin{cases}
     1, & e_i + c_i + \sum_{j \in \mathrm{pa}(i) }{b_{ij} x_j}>0 \\
     0, & \mathrm{otherwise}
    \end{cases}, \quad e_i \sim Logistic(0,1),
   \end{eqnarray}

where the error terms :math:`e_i` follow the Logistic distribution, while the other notations are identical to those in continuous variables. 

This method makes the following assumptions. 

#. Continous variables and binary variables.
#. Linearity
#. Acyclicity
#. No hidden common causes
#. Baselines are the same when predicting one binary variable from the other for every pair of binary variables.

References

    .. [1] Y. Zeng, S. Shimizu, H. Matsui, F. Sun. 
       Causal discovery for linear mixed data. 
       In Proc. First Conference on Causal Learning and Reasoning (CLeaR2022). PMLR 177, pp. 994-1009, 2022.  

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
