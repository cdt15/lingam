
LiNA
========

Model
-------------------
LiNA [1]_ allows to locate the latent factors as well as uncover the causal structure between such latent factors of interests. 
Causal structure between latent factors can be ubiquitous in real-world applications, e.g., relations bewteen anxiety, depression, and coping in psychology [2]_ [3]_ , etc.

This method is based on the LiNA model as shown below, 

.. math::
    
    {f}^{(m)}&= {B}^{(m)} {f}^{(m)}+{\varepsilon}^{(m)}, \\
    {x}^{(m)}&={G}^{(m)}{f}^{(m)}+{e}^{(m)},

where :math:`{\varepsilon}^{(m)}` and :math:`{e}^{(m)}` are random vectors that collect external influences, and errors, respectively, 
and they are independent with each other. :math:`{f}^{(m)}` and :math:`{x}^{(m)}` are random vectors that collect latent factors, and observed data, respectively.
:math:`{B}^{(m)}` is a matrix that collects causal effects :math:`b_{ij}^{(m)}` between :math:`{f}_i^{(m)}` and :math:`{f}_j^{(m)}`, 
while :math:`{G}^{(m)}` collects factor loadings :math:`g_{ij}^{(m)}` between :math:`{f}_j^{(m)}` and :math:`{x}_i^{(m)}`. :math:`m`
stands for the :math:`m^{th}` domain.

This method makes the following assumptions. 

#. Linearity
#. Acyclicity
#. No causal relations between observed variables
#. Non-Gaussian continuous distubance variables (except at most one) for latent factors
#. Gaussian error variables (except at most one) for observed variables
#. Each latent factor has at lest 2 pure measurement variables.

References

    .. [1] Y. Zeng, S. Shimizu, R. Cai, F. Xie, M. Yamamoto, and Z. Hao. 
        Causal discovery with multi-domain LiNGAM for latent factors. 
        In Proc. Thirtieth International Joint Conference on Artificial Intelligence (IJCAI-21), 2021.
    .. [2] R. Silva, R. Scheines, C. Glymour, and P. Spirtes.
       Learning the structure of linear latent variable models.
       Journal of Machine Learning Research, 7(2):191-246, 2006.
    .. [3] D. Bartholomew, F. Steele, I. Moustaki, and J. Galbraith.
       The analysis and interpretation of multivariate data for social scientists.
       Routledge (Second edition), 2008.


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

    ['1.20.3', '1.5.4']
    

Single-domain test data 
-------------------------

First, we generate a causal structure with 10 measurement variables and 5 latent factors, where each latent
variable has 2 pure measurement variables.

.. code-block:: python

    ut.set_random_seed(1)
    noise_ratios = 0.1
    n_features = 10  # number of measurement vars.
    n_samples, n_features_latent, n_edges, graph_type, sem_type = 1000, 5, 5, 'ER', 'laplace'
    B_true = ut.simulate_dag(n_features_latent, n_edges, graph_type)
    W_true = ut.simulate_parameter(B_true)  # row to column

    f, E, E_weight = ut.simulate_linear_sem(W_true, n_samples, sem_type)
    f_nor = np.zeros([n_samples, n_features_latent])
    scale = np.zeros([1, n_features_latent])
    W_true_scale = np.zeros([n_features_latent, n_features_latent])
    for j in range(n_features_latent):
        scale[0, j] = np.std(f[:, j])
        f_nor[:, j] = f[:, j] / np.std(f[:, j])
        W_true_scale[:, j] = W_true[:, j] / scale[0, j]  # scaled W_true

    # generate noises ei of xi
    e = np.random.random([n_features, n_samples])
    for j in range(n_features):
        e[j, :] = e[j, :] - np.mean(e[j, :])
        e[j, :] = e[j, :] / np.std(e[j, :])

    G = np.zeros([n_features, n_features_latent])
    G[0, 0] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G[1, 0] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G[2, 1] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G[3, 1] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G[4, 2] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G[5, 2] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G[6, 3] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G[7, 3] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G[8, 4] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G[9, 4] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G_sign = np.sign(G)

    # normalize G
    G_nor = np.zeros([n_features, n_features_latent])
    for j in range(n_features):
        e[j, :] = e[j, :] / np.sqrt(np.square(np.sum(G[j, :])) + np.square(noise_ratios))
        G_nor[j, :] = G[j, :] / np.sqrt(np.square(np.sum(G[j, :])) + np.square(noise_ratios))

    X = G_nor @ f_nor.T + noise_ratios * e  # X:n_features*n_samples   "e is small or n_features are large"
    X = X.T

    print('The true adjacency matrix is:\n', W_true)

   
.. parsed-literal::

    The true adjacency matrix is:
    [[ 0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.52905044 -1.87243368]
     [-1.94141783  0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          1.12108398]
     [ 0.          0.         -0.87478353  0.          0.        ]]



Causal Discovery for single-domain data
-----------------------------------------

To run causal discovery, we create a ``LiNA`` object and call the ``fit``
method.

.. code-block:: python

    model = lingam.LiNA()
    model.fit(X, G_sign, scale)



.. parsed-literal::

    <lingam.lina.LiNA at 0x2130f482970>



Using the ``_adjacency_matrix`` properties, we can see the estimated adjacency 
matrix between latent factors.



.. code-block:: python

    print('The estimated adjacency matrix is:\n', model._adjacency_matrix)



.. parsed-literal::

    The estimated adjacency matrix is:
    [[ 0.          0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.51703777 -1.75584025]
     [-1.75874721  0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.99860274]
     [ 0.          0.         -0.77518384  0.          0.        ]]






Multi-domain test data 
-------------------------

We generate a causal structure with 2 domains where in each domain there are 6 measurement variables and 3 latent factors. Each latent factor has 2 pure measurement variables. 


.. code-block:: python

    n_features = 6  # number of measurement vars. in each domain
    noise_ratios = 0.1

    ut.set_random_seed(1)

    n_samples, n_features_latent, n_edges, graph_type, sem_type1, sem_type2 = 1000, 3, 3, 'ER', 'subGaussian', 'supGaussian'
    # n_edges: number of edges btw. latent factors in a domain
    # sem_type1/sem_type2: different distributions of noises from different domains
    B_true = ut.simulate_dag(n_features_latent, n_edges, graph_type)  # skeleton btw. latent factors
    W_true = ut.simulate_parameter(B_true)  # causal effects matrix btw. latent factors

    # 1 domain
    f, E, E_weight = ut.simulate_linear_sem(W_true, n_samples, sem_type1)
    f_nor1 = np.zeros([n_samples, n_features_latent])
    scale1 = np.zeros([1, n_features_latent])
    W_true_scale = np.zeros([n_features_latent, n_features_latent])
    for j in range(n_features_latent):
        scale1[0, j] = np.std(f[:, j])
        f_nor1[:, j] = f[:, j] / np.std(f[:, j])
        W_true_scale[:, j] = W_true[:, j] / scale1[0, j]
    e = np.random.random([n_features, n_samples])
    for j in range(n_features):
        e[j, :] = e[j, :] - np.mean(e[j, :])
        e[j, :] = e[j, :] / np.std(e[j, :])

    G1 = np.zeros([n_features, n_features_latent])
    G1[0, 0] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G1[1, 0] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G1[2, 1] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G1[3, 1] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G1[4, 2] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G1[5, 2] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G_sign1 = np.sign(G1)
    # normalize G
    G_nor1 = np.zeros([n_features, n_features_latent])
    for j in range(n_features):
        e[j, :] = e[j, :] / np.sqrt(np.square(np.sum(G1[j, :])) + np.square(noise_ratios))
        G_nor1[j, :] = G1[j, :] / np.sqrt(np.square(np.sum(G1[j, :])) + np.square(noise_ratios))
    X1 = G_nor1 @ f_nor1.T + noise_ratios * e  # "the noise ratio e is small or n_features is large"
    X1 = X1.T

    # 2 domain
    f2, E, E_weight = ut.simulate_linear_sem(W_true, n_samples, sem_type2)
    f_nor2 = np.zeros([n_samples, n_features_latent])
    scale2 = np.zeros([1, n_features_latent])
    W_true_scale = np.zeros([n_features_latent, n_features_latent])
    for j in range(n_features_latent):
        scale2[0, j] = np.std(f2[:, j])
        f_nor2[:, j] = f2[:, j] / np.std(f2[:, j])
        W_true_scale[:, j] = W_true[:, j] / scale2[0, j]
    e = np.random.random([n_features, n_samples])
    for j in range(n_features):
        e[j, :] = e[j, :] - np.mean(e[j, :])
        e[j, :] = e[j, :] / np.std(e[j, :])
    G2 = np.zeros([n_features, n_features_latent])
    G2[0, 0] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G2[1, 0] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G2[2, 1] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G2[3, 1] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G2[4, 2] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G2[5, 2] = random.choice((-1, 1)) * (0.2 + 0.5 * np.random.rand(1))
    G_sign2 = np.sign(G2)
    # normalize G 
    G_nor2 = np.zeros([n_features, n_features_latent])
    for j in range(n_features):
        e[j, :] = e[j, :] / np.sqrt(np.square(np.sum(G2[j, :])) + np.square(noise_ratios))
        G_nor2[j, :] = G2[j, :] / np.sqrt(np.square(np.sum(G2[j, :])) + np.square(noise_ratios))
    X2 = G_nor2 @ f_nor2.T + noise_ratios * e
    X2 = X2.T  # X:n_samples * n_features

    # augment the data X
    X = scipy.linalg.block_diag(X1, X2)
    G_sign = scipy.linalg.block_diag(G_sign1, G_sign2)
    scale = scipy.linalg.block_diag(scale1, scale2)

    print('The true adjacency matrix is:\n', W_true)

.. parsed-literal::

    The true adjacency matrix is:
    [[0.         1.18580721 1.14604785]
     [0.         0.         0.        ]
     [0.         0.63920121 0.        ]]

   


Causal Discovery for multi-domain data
---------------------------------------------

To run causal discovery, we create a ``MDLiNA`` object and call the ``fit``
method.

.. code-block:: python

    model = lingam.MDLiNA()
    model.fit(XX, G_sign, scale)



.. parsed-literal::

    <lingam.lina.MDLiNA at 0x1812ee2fdf0>



Using the ``_adjacency_matrix`` properties, we can see the estimated adjacency 
matrix between latent factors of interest.



.. code-block:: python

    print('The estimated adjacency matrix is:\n', model._adjacency_matrix)



.. parsed-literal::

    The estimated adjacency matrix is:
    [[ 0.          0.34880702 -0.78706636]
     [ 0.          0.          0.61577239]
     [ 0.          0.          0.        ]]
