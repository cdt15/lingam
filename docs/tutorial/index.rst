Tutorial
========
In this tutorial, we will show you how to run LiNGAM algorithms and see the results. We will also show you how to run the bootstrap method and check the results.

The following packages must be installed in order to run this tutorial. And import if necessary:

* numpy
* pandas
* scikit-learn
* graphviz
* statsmodels

Contents:

Basic DAG models
----------------

.. toctree::
    :maxdepth: 1

    lingam
    bootstrap
    bootstrap_with_imputation
    pk_direct
    directlingam_gpu
    resit
    pk_resit
    group_direct_lingam
    high_dim_direct_lingam

Time Series
-----------

.. toctree::
    :maxdepth: 1

    var
    varma
    longitudinal
    longitudinal_resit

Multiple datasets
-----------------

.. toctree::
    :maxdepth: 1

    multiple_dataset
    multi_group_rcd
    multi_group_camuv
    multi_group_resit

Latent confounders and latent factors
-------------------------------------

.. toctree::
    :maxdepth: 1

    bottom_up_parce
    pk_bottom_up_parce
    rcd
    camuv
    lina
    group_lingam
    abic_lingam

Mixed data
----------

.. toctree::
    :maxdepth: 1

    lim

Missing data
------------

.. toctree::
    :maxdepth: 1

    missingness_lingam

Tools
-----

.. toctree::
    :maxdepth: 1

    total_effect
    causal_effect
    extract_ancestors
    f_correlation
    draw_graph
    visualize_nonlinear_causal_effect
    evaluate_model_fit
    lewis
