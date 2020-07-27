Total Effect
============

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
