import numpy as np

from lingam.tools import (
    bootstrap_with_imputation,
)


def test_bootstrap_with_imputation():
    # test data
    sample_size = 1000
    
    m = np.array([
        [ 0.000,  0.000,  0.000,  0.895,  0.000,  0.000],
        [ 0.565,  0.000,  0.377,  0.000,  0.000,  0.000],
        [ 0.000,  0.000,  0.000,  0.895,  0.000,  0.000],
        [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000],
        [ 0.991,  0.000, -0.124,  0.000,  0.000,  0.000],
        [ 0.895,  0.000,  0.000,  0.000,  0.000,  0.000]
    ])
    
    error_vars = [0.2, 0.2, 0.2, 1.0, 0.2, 0.2]
    params = [0.5 * np.sqrt(12 * v) for v in error_vars]
    
    generate_error = lambda p: np.random.uniform(-p, p, size=sample_size)
    e = np.array([generate_error(p) for p in params])
    
    X = np.linalg.pinv(np.eye(len(m)) - m) @ e
    X = X.T

    # test data with missing values by MCAR
    X_mcar = X.copy()
    
    prop_missing = [0, 0, 0, 0, 0, 0.05]
    for i, prop in enumerate(prop_missing):
        mask = np.random.uniform(0, 1, size=len(X_mcar))
        X_mcar[mask < prop, i] = np.nan

    # default
    n_sampling = 5
    n_repeats = 3
    causal_orders, adj_matrices_list, resampled_indices, imputation_results = bootstrap_with_imputation(
        X_mcar, n_sampling, n_repeats, random_state=0)

    # check shapes
    assert causal_orders.shape == (n_sampling, len(m))
    assert adj_matrices_list.shape == (n_sampling, n_repeats, *m.shape)
    assert resampled_indices.shape == (n_sampling, sample_size)
    assert imputation_results.shape == (n_sampling, n_repeats, *X_mcar.shape)

    # check causal orders
    assert np.all([np.unique(r) == np.arange(len(m)) for r in causal_orders])

    # confirm imputed positions
    for i, imputation_result in enumerate(imputation_results):
        indices = resampled_indices[i]
        resampled = X_mcar[indices]
        nan_pos = np.argwhere(np.isnan(resampled))

        for result in imputation_result:
            imputation_pos = np.argwhere(~np.isnan(result))
            assert (nan_pos == imputation_pos).all()

