import numpy as np

from lingam.tools import (
    bootstrap_with_imputation,
    BaseMultipleImputation,
    BaseMultiGroupCDModel,
)


def test_bootstrap_with_imputation():
    # test data
    m = np.array([
        [ 0.000,  0.000,  0.000,  0.895,  0.000,  0.000],
        [ 0.565,  0.000,  0.377,  0.000,  0.000,  0.000],
        [ 0.000,  0.000,  0.000,  0.895,  0.000,  0.000],
        [ 0.000,  0.000,  0.000,  0.000,  0.000,  0.000],
        [ 0.991,  0.000, -0.124,  0.000,  0.000,  0.000],
        [ 0.895,  0.000,  0.000,  0.000,  0.000,  0.000]
    ])
    error_vars = [0.2, 0.2, 0.2, 1.0, 0.2, 0.2]
    n_samples = 1000
    X = _generate_data(m, error_vars, sample_size=n_samples)

    # test data with missing values by MCAR
    prop_list = [0, 0, 0, 0, 0, 0.05]
    X_mcar = _add_nan_mcar(X, prop_list=prop_list)
    
    # default
    n_sampling = 5
    n_repeats = 3
    causal_orders, adj_matrices_list, resampled_indices, imputation_results = bootstrap_with_imputation(
        X_mcar, n_sampling, n_repeats, random_state=0)

    # check shapes
    assert causal_orders.shape == (n_sampling, len(m))
    assert adj_matrices_list.shape == (n_sampling, n_repeats, *m.shape)
    assert resampled_indices.shape == (n_sampling, n_samples)
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


def test_bootstrap_with_imputation_imp():
    """ tests for the function to check the output of the given imputer. """

    # test data
    m = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ]).astype(float)
    error_vars = np.ones(3)
    n_samples = 10
    X_org = _generate_data(m, error_vars, sample_size=n_samples)
    X_mcar = _add_nan_mcar(X_org, prop_list=None)

    n_features = len(m)
    n_sampling = 1
    n_repeats = 2

    # valid
    class ValidImputer(BaseMultipleImputation):
        def fit_transform(self, X):
            imputeds = np.array([X_org for _ in range(n_repeats)])
            return imputeds

    imp = ValidImputer()
    result = bootstrap_with_imputation(X_mcar, n_sampling, imp=imp)

    causal_order, adj_matrices_list, resampled_indices, imputation_results = result
    assert causal_order.shape == (n_sampling, len(m))
    assert adj_matrices_list.shape == (n_sampling, n_repeats, *m.shape)
    assert resampled_indices.shape == (n_sampling, n_samples)
    assert imputation_results.shape == (n_sampling, n_repeats, *X_mcar.shape)

    # invalid: not inherited
    class InvalidImputer:
        def fit_transform(self, X):
            imputeds = np.array([X_org for _ in range(n_repeats)])
            return imputeds

    imp = InvalidImputer()
    try:
        bootstrap_with_imputation(X_mcar, n_sampling, imp=imp)
    except Exception as e:
        pass
    else:
        raise AssertionError

    # invalid: exists nan 
    class InvalidImputer(BaseMultipleImputation):
        def fit_transform(self, X):
            imputeds = np.array([X_mcar for _ in range(n_repeats)])
            return imputeds

    imp = InvalidImputer()
    try:
        bootstrap_with_imputation(X_mcar, n_sampling, imp=imp)
    except Exception as e:
        pass
    else:
        raise AssertionError

    # invalid: wrong shape
    class InvalidImputer(BaseMultipleImputation):
        def fit_transform(self, X):
            imputeds = np.array([X_org for _ in range(n_repeats)])
            imputeds = imputeds[:, :-1, :]
            return imputeds

    imp = InvalidImputer()
    try:
        bootstrap_with_imputation(X_mcar, n_sampling, imp=imp)
    except Exception as e:
        pass
    else:
        raise AssertionError

    # invalid: wrong data type
    class InvalidImputer(BaseMultipleImputation):
        def fit_transform(self, X):
            imputeds = np.array([X_org for _ in range(n_repeats)])
            return imputeds.astype(str)

    imp = InvalidImputer()
    try:
        bootstrap_with_imputation(X_mcar, n_sampling, imp=imp)
    except Exception as e:
        pass
    else:
        raise AssertionError

def test_bootstrap_with_imputation_cd_model():
    """ tests for the function to check the output of the given cd_model """

    # test data
    m = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ]).astype(float)
    error_vars = np.ones(3)
    n_samples = 10
    X = _generate_data(m, error_vars, sample_size=n_samples)
    X_mcar = _add_nan_mcar(X, prop_list=None)

    n_features = len(m)
    n_sampling = 1
    n_repeats = 2

    # valid
    class ValidCDModel(BaseMultiGroupCDModel):
        def before_imputation(self, X):
            pass
        def fit(self, X_list):
            co = [0, 1, 2]
            ams = [m for _ in range(len(X_list))]
            return co, ams

    n_sampling = 2
    cd_model = ValidCDModel()
    result = bootstrap_with_imputation(X_mcar, n_sampling, n_repeats=n_repeats, cd_model=cd_model)

    causal_order, adj_matrices_list, resampled_indices, imputation_results = result
    assert causal_order.shape == (n_sampling, len(m))
    assert adj_matrices_list.shape == (n_sampling, n_repeats, *m.shape)
    assert resampled_indices.shape == (n_sampling, n_samples)
    assert imputation_results.shape == (n_sampling, n_repeats, *X_mcar.shape)

    # valid: cd_model estimates one adj_matrix based on X_list
    class ValidCDModel(BaseMultiGroupCDModel):
        def before_imputation(self, X):
            pass
        def fit(self, X_list):
            co = [0, 1, 2]
            ams = m.copy()
            return co, ams

    n_sampling = 2
    cd_model = ValidCDModel()
    result = bootstrap_with_imputation(X_mcar, n_sampling, n_repeats=n_repeats, cd_model=cd_model)

    causal_order, adj_matrices_list, resampled_indices, imputation_results = result
    assert causal_order.shape == (n_sampling, len(m))
    assert adj_matrices_list.shape == (n_sampling, *m.shape)
    assert resampled_indices.shape == (n_sampling, n_samples)
    assert imputation_results.shape == (n_sampling, n_repeats, *X_mcar.shape)

    # valid: adj_matrices: contains nan
    class ValidCDModel(BaseMultiGroupCDModel):
        def before_imputation(self, X):
            pass
        def fit(self, X_list):
            co = [0, 1, 2]
            ams = [m.copy() for _ in range(len(X_list))]
            ams[0][0, 0] = np.nan
            return (co, ams)

    n_sampling = 2
    cd_model = ValidCDModel()
    result = bootstrap_with_imputation(X_mcar, n_sampling, n_repeats=n_repeats, cd_model=cd_model)
    
    causal_order, adj_matrices_list, resampled_indices, imputation_results = result
    assert causal_order.shape == (n_sampling, len(m))
    assert adj_matrices_list.shape == (n_sampling, n_repeats, *m.shape)
    assert resampled_indices.shape == (n_sampling, n_samples)
    assert imputation_results.shape == (n_sampling, n_repeats, *X_mcar.shape)

    # invalid: not inherited
    class InvalidCDModel:
        def before_imputation(self, X):
            pass
        def fit(self, X_list):
            co = [0, 1, 2]
            ams = [m for _ in range(len(X_list))]
            return co, ams

    cd_model = InvalidCDModel()
    try:
        bootstrap_with_imputation(X_mcar, n_sampling, n_repeats=n_repeats, cd_model=cd_model)
    except Exception as e:
        pass
    else:
        raise AssertionError

    # invalid: wrong type
    class InvalidCDModel(BaseMultiGroupCDModel):
        def before_imputation(self, X):
            pass
        def fit(self, X_list):
            co = [0, 1, 2]
            ams = [m for _ in range(len(X_list))]
            return [co, ams]

    cd_model = InvalidCDModel()
    try:
        bootstrap_with_imputation(X_mcar, n_sampling, n_repeats=n_repeats, cd_model=cd_model)
    except Exception as e:
        pass
    else:
        raise AssertionError

    # invalid: wrong length
    class InvalidCDModel(BaseMultiGroupCDModel):
        def before_imputation(self, X):
            pass
        def fit(self, X_list):
            co = [0, 1, 2]
            ams = [m for _ in range(len(X_list))]
            return (co, ams, 0)

    cd_model = InvalidCDModel()
    try:
        bootstrap_with_imputation(X_mcar, n_sampling, n_repeats=n_repeats, cd_model=cd_model)
    except Exception as e:
        pass
    else:
        raise AssertionError

    # invalid: causal_order: contains nan
    class InvalidCDModel(BaseMultiGroupCDModel):
        def before_imputation(self, X):
            pass
        def fit(self, X_list):
            co = [np.nan, 1, 2]
            ams = [m for _ in range(len(X_list))]
            return (co, ams)

    cd_model = InvalidCDModel()
    try:
        bootstrap_with_imputation(X_mcar, n_sampling, n_repeats=n_repeats, cd_model=cd_model)
    except Exception as e:
        pass
    else:
        raise AssertionError

    # invalid: causal_order: wrong length
    class InvalidCDModel(BaseMultiGroupCDModel):
        def before_imputation(self, X):
            pass
        def fit(self, X_list):
            co = [0, 1, 2, 3]
            ams = [m for _ in range(len(X_list))]
            return (co, ams)

    cd_model = InvalidCDModel()
    try:
        bootstrap_with_imputation(X_mcar, n_sampling, n_repeats=n_repeats, cd_model=cd_model)
    except Exception as e:
        pass
    else:
        raise AssertionError

    # invalid: causal_order: not unique
    class InvalidCDModel(BaseMultiGroupCDModel):
        def before_imputation(self, X):
            pass
        def fit(self, X_list):
            co = [0, 0, 1]
            ams = [m for _ in range(len(X_list))]
            return (co, ams)

    cd_model = InvalidCDModel()
    try:
        bootstrap_with_imputation(X_mcar, n_sampling, n_repeats=n_repeats, cd_model=cd_model)
    except ValueError as e:
        pass
    else:
        raise AssertionError

    # invalid: adj_matrices: wrong type
    class InvalidCDModel(BaseMultiGroupCDModel):
        def before_imputation(self, X):
            pass
        def fit(self, X_list):
            co = [0, 1, 2]
            ams = [m.copy().astype(str) for _ in range(len(X_list))]
            return (co, ams)

    cd_model = InvalidCDModel()
    try:
        bootstrap_with_imputation(X_mcar, n_sampling, n_repeats=n_repeats, cd_model=cd_model)
    except Exception as e:
        pass
    else:
        raise AssertionError

    # invalid: adj_matrices: wrong shape
    class InvalidCDModel(BaseMultiGroupCDModel):
        def before_imputation(self, X):
            pass
        def fit(self, X_list):
            co = [0, 1, 2]
            ams = [m.copy()[:-1, :] for _ in range(len(X_list))]
            return (co, ams)

    cd_model = InvalidCDModel()
    try:
        bootstrap_with_imputation(X_mcar, n_sampling, n_repeats=n_repeats, cd_model=cd_model)
    except Exception as e:
        pass
    else:
        raise AssertionError

def _generate_data(m, error_vars, sample_size=1000):
    params = [0.5 * np.sqrt(12 * v) for v in error_vars]
    generate_error = lambda p: np.random.uniform(-p, p, size=sample_size)
    e = np.array([generate_error(p) for p in params])
    
    X = np.linalg.pinv(np.eye(len(m)) - m) @ e
    X = X.T

    return X

def _add_nan_mcar(X, prop_list=None):
    X_mcar = X.copy()

    if prop_list is None:
        prop_list = np.ones(X_mcar.shape[1]) * 0.1

    for i, prop in enumerate(prop_list):
        mask = np.random.uniform(0, 1, size=len(X_mcar))
        X_mcar[mask < prop, i] = np.nan

    return X_mcar


