import numpy as np
import pytest

from lingam import VARLiNGAM


def _model(coefficients, order):
    model = VARLiNGAM(lags=len(coefficients) - 1)
    model._adjacency_matrices = np.asarray(coefficients)
    model._causal_order = list(order)
    return model


def _block_response(coefficients, horizon):
    """Solve all structural equations in chronological order at once."""
    n_features = len(coefficients[0])
    size = n_features * (horizon + 1)
    system = np.eye(size)
    for time in range(horizon + 1):
        rows = slice(time * n_features, (time + 1) * n_features)
        for lag, coefficient in enumerate(coefficients[: time + 1]):
            cols = slice((time - lag) * n_features, (time - lag + 1) * n_features)
            system[rows, cols] -= coefficient
    impulse = np.zeros((size, n_features))
    impulse[:n_features] = np.eye(n_features)
    return np.linalg.solve(system, impulse)[-n_features:]


@pytest.mark.parametrize(("first", "second", "expected"), [(0.5, 0.8, 0.4), (-0.6, -0.5, 0.3)])
def test_total_effect_through_two_lag_one_edges(first, second, expected):
    coefficients = np.zeros((3, 3, 3))
    coefficients[1, 1, 0] = first
    coefficients[1, 2, 1] = second
    model = _model(coefficients, [0, 1, 2])

    assert model.estimate_total_effect2(3, 0, 2, from_lag=2) == pytest.approx(expected)
    assert model.estimate_total_effect2(3, 2, 0, from_lag=2) == 0.0
    assert model.estimate_total_effect2(3, 0, 2, from_lag=1) == 0.0


@pytest.fixture
def coefficients():
    matrices = np.zeros((4, 4, 4))
    # Instantaneous order is 2 -> 0 -> 1, with a direct route 2 -> 1.
    # Variable 3 is disconnected, and lagged edges need not follow that order.
    matrices[0, 0, 2] = 0.6
    matrices[0, 1, 0] = -0.25
    matrices[0, 1, 2] = 0.1
    matrices[1, :3, :3] = [[0.2, -0.2, 0.0], [0.0, 0.1, 0.25], [0.3, 0.0, 0.15]]
    matrices[2, 1, 2] = -0.2
    matrices[2, 0, 0] = 0.1
    matrices[3, 2, 1] = 0.12
    return matrices


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_lagged_effects_match_structural_block_system(coefficients, horizon):
    model = _model(coefficients, [2, 0, 1, 3])
    expected = _block_response(coefficients, horizon)
    actual = np.array(
        [
            [model.estimate_total_effect2(4, source, target, horizon) for source in range(4)]
            for target in range(4)
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(actual[3, :], 0)
    np.testing.assert_array_equal(actual[:, 3], 0)


def test_instantaneous_paths_and_causal_order_are_preserved(coefficients):
    model = _model(coefficients, [2, 0, 1, 3])
    assert model.estimate_total_effect2(4, 2, 1) == pytest.approx(0.1 + 0.6 * -0.25)
    assert model.estimate_total_effect2(4, 2, 3) == 0.0
    with pytest.warns(UserWarning, match="earlier than the source"):
        assert model.estimate_total_effect2(4, 1, 2) == 0.0
    with pytest.raises(ValueError, match="mustn't be the same"):
        model.estimate_total_effect2(4, 2, 2)


@pytest.mark.parametrize(
    ("source", "target", "horizon"), [(0, -1, 1), (0, 1, 4), (-1, 1, 1)]
)
def test_lagged_indices_do_not_alias_other_variables(coefficients, source, target, horizon):
    model = _model(coefficients, [2, 0, 1, 3])
    with pytest.raises(ValueError):
        model.estimate_total_effect2(4, source, target, horizon)


def test_bootstrap_lag_blocks_match_their_fitted_structural_models():
    ar = np.array([[[0.4, -0.1], [0.3, 0.2]], [[-0.08, 0.15], [0.12, 0.02]]])
    rng = np.random.default_rng(17)
    data = rng.laplace(size=(220, 2))
    for t in range(2, len(data)):
        data[t] += ar[0] @ data[t - 1] + ar[1] @ data[t - 2]
    model = VARLiNGAM(lags=2, ar_coefs=ar, prune=False, random_state=17)
    result = model.bootstrap(data, n_sampling=2)

    assert result.total_effects_.shape == (2, 2, 6)
    for adjacency, actual in zip(result.adjacency_matrices_, result.total_effects_):
        matrices = np.split(adjacency, 3, axis=1)
        instantaneous = _block_response(matrices, 0)
        np.fill_diagonal(instantaneous, 0)
        expected = np.concatenate(
            [instantaneous, _block_response(matrices, 1), _block_response(matrices, 2)],
            axis=1,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
