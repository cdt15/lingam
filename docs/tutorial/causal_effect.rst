Causal Effect on predicted variables
=============

The following demonstrates a method [1]_ that analyzes the prediction mechanisms of constructed predictive models based on causality.
This method estimates causal effects, i.e., intervention effects of features or explnatory variables used in constructed predictive models on the predicted variables. 
Users can use estimated causal structures, e.g., by a LiNGAM-type method or known causal structures based on domain knowledge. 

References

    .. [1] P. Blöbaum and S. Shimizu. Estimation of interventional effects of features on prediction. 
      In Proc. 2017 IEEE International Workshop on Machine Learning for Signal Processing (MLSP2017), pp. 1--6, Tokyo, Japan, 2017.


First, we use lingam package:

.. code-block:: python

    import lingam

Then, we create a :class:`~lingam.DirectLiNGAM` object and call the :func:`~lingam.DirectLiNGAM.fit` method:

.. code-block:: python

    model = lingam.DirectLiNGAM()
    model.fit(X)

Next, we create the prediction model. In the following example, linear regression model is created, but it is also possible to create logistic regression model or non-linear regression model.

.. code-block:: python

    from sklearn.linear_model import LinearRegression

    target = 0
    features = [i for i in range(X.shape[1]) if i != target]
    reg = LinearRegression()
    reg.fit(X.iloc[:, features], X.iloc[:, target])


Identification of Feature with Greatest Causal Influence on Prediction
----------------------------------------------------------------------

We create a :class:`~lingam.CausalEffect` object and call the :func:`~lingam.CausalEffect.estimate_effects_on_prediction` method.

.. code-block:: python

    ce = lingam.CausalEffect(model)
    effects = ce.estimate_effects_on_prediction(X, target, reg)

To identify of the feature having the greatest intervention effect on the prediction, we can get the feature that maximizes the value of the obtained list.

.. code-block:: python

    print(X.columns[np.argmax(effects)])

.. code-block:: python

    cylinders

Estimation of Optimal Intervention
----------------------------------

To estimate of the intervention such that the expectation of the prediction of the post-intervention observations is equal or close to a specified value, we use :func:`~lingam.CausalEffect.estimate_optimal_intervention` method of :class:`~lingam.CausalEffect`.
In the following example, we estimate the intervention value at variable index 1 so that the predicted value is close to 15.

.. code-block:: python

    c = ce.estimate_optimal_intervention(X, target, reg, 1, 15)
    print(f'Optimal intervention: {c:.3f}')

.. code-block:: python

    Optimal intervention: 7.871

Use a known causal model
------------------------

When using a known causal model, we can specify the adjacency matrix when we create :class:`~lingam.CausalEffect` object.

.. code-block:: python

    m = np.array([[0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                  [3.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [8.0, 0.0,-1.0, 0.0, 0.0, 0.0],
                  [4.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    
    ce = lingam.CausalEffect(causal_model=m)
    effects = ce.estimate_effects_on_prediction(X, target, reg)

For details, see also:

* https://github.com/cdt15/lingam/blob/master/examples/CausalEffect.ipynb
* https://github.com/cdt15/lingam/blob/master/examples/CausalEffect(LassoCV).ipynb
* https://github.com/cdt15/lingam/blob/master/examples/CausalEffect(LogisticRegression).ipynb
* https://github.com/cdt15/lingam/blob/master/examples/CausalEffect(LightGBM).ipynb
