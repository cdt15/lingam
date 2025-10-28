m-LiNGAM
=========

Model
-------------------

Missingness-LiNGAM (m-LiNGAM) [1]_ extends the basic LiNGAM model [2]_ to handle datasets affected by missing values, including Missing Completely At Random (MCAR), Missing At Random (MAR), and Missing Not At Random (MNAR) cases.  
It enables the identification of the true underlying causal structure and provides unbiased parameter estimates even when data are not fully observed.

The model combines the principles of LiNGAM and the graphical representation of missingness mechanisms using *missingness graphs* (m-graphs) [3]_.  
In this framework, variables can be fully observed or partially observed, and each partially observed variable is associated with a missingness mechanism and a proxy variable.  

Let the set of variables be:

.. math::

   V = V_o \cup V_m \cup U \cup V^* \cup R

where: 

- :math:`V_o` are fully observed variables,  
- :math:`V_m` are partially observed variables,  
- :math:`U` are latent variables (here assumed empty),  
- :math:`V^*` are proxy variables (what is actually observed, corresponding to dataset columns with missing values)
- :math:`R` are missingness mechanism.

The induced subgraph :math:`G[V_o \cup V_m]` follows a LiNGAM model, meaning that for every variable :math:`X_i \in (V_o \cup V_m)`:

.. math::

  x_i = \sum_{k(j)<k(i)}b_{ij}x_j + e_i, \qquad e_i\sim \text{Non-Gaussian}(\cdot)

where :math:`i\in\{1,\dots,n\}\mapsto k(i)` denotes a causal order, and the non-gaussian error terms are independent.

The induced subgraph :math:`G[V_o \cup V_m \cup R]` follows a LiM model. The missingness mechanisms :math:`R_i \in R` follow a logistic model as for binary variables in LiM [4]_:

.. math::
    x_i = \mathbf 1\llbracket\sum_{k(j)<k(i)} b_{ij} x_j + e_i > 0\rrbracket, \qquad e_i \sim \text{Logistic}(0,1)


Assumptions
^^^^^^^^^^^^^^^^^^

The following assumptions are made to ensure identifiability:

#. No latent confounders (:math:`U = \emptyset`).
#. No causal interactions between missingness mechanisms (:math:`R_i \notin Pa(R_j)` for all :math:`i \neq j`).
#. No self-masking (:math:`X_i \notin Pa(R_i)` for any :math:`X_i \in V_m`).

Note that even if self-masking is not allowed, indirect self-masking is: a partially observed variable can be an indirect cause (an ancestor) of its own missingness mechanism.
Under these assumptions, m-LiNGAM guarantees identifiability of both the causal structure and parameters from observational data in the large-sample limit.

An example Python notebook demonstrating m-LiNGAM is available `here <https://github.com/cdt15/lingam/blob/master/examples/MissingnessLiNGAM.ipynb>`__.

References
-------------------

.. [1] M. Ceriscioli, S. Shimizu, and K. Mohan.  
       *Discovering Linear Non-Gaussian Models for All Categories of Missing Data (Student Abstract).*  
       The 40th Annual AAAI Conference on Artificial Intelligence (AAAI-26) Student Abstract and Poster Program, 2026.

.. [2] S. Shimizu, P. O. Hoyer, A. Hyvärinen, and A. J. Kerminen.  
       *A Linear Non-Gaussian Acyclic Model for Causal Discovery.*  
       Journal of Machine Learning Research, 7:2003–2030, 2006.

.. [3] K. Mohan, J. Pearl, and J. Tian.  
       *Graphical Models for Inference with Missing Data.*  
       Advances in Neural Information Processing Systems (NeurIPS), 2013.

.. [4] Y. Zeng, S. Shimizu, H. Matsui, and F. Sun.  
       *Causal Discovery for Linear Mixed Data.*  
       In Proceedings of the First Conference on Causal Learning and Reasoning (CLeaR 2022), PMLR 177, pp. 994–1009, 2022.