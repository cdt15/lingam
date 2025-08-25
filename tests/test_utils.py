import random

import numpy as np
import pandas as pd

from lingam.utils import (
    print_causal_directions,
    print_dagc,
    make_prior_knowledge,
    remove_effect,
    make_dot,
    make_dot_highlight,
    predict_adaptive_lasso,
    get_sink_variables,
    get_exo_variables,
    find_all_paths,
    extract_ancestors,
    f_correlation,
    evaluate_model_fit,
    calculate_distance_from_root_nodes,
    calculate_total_effect,
    get_common_edge_probabilities,
    print_common_edge_directions,
    make_dot_for_nan_probability_matrix,
)


def test_print_causal_directions():
    cdc = {"from": [0, 1, 2, 3], "to": [1, 2, 3, 4], "count": [5, 4, 3, 2]}
    print_causal_directions(cdc, n_sampling=5)
    print_causal_directions(cdc, n_sampling=5, labels=["A", "B", "C", "D", "E"])


def test_print_dagc():
    dagc = {
        "dag": [
            {"from": [0, 1, 2, 3], "to": [1, 2, 3, 4]},
            {"from": [0, 1, 2, 3], "to": [1, 2, 3, 0]},
            {"from": [0, 1, 2, 3], "to": [1, 2, 3, 4]},
        ],
        "count": [5, 2, 1],
    }
    print_dagc(dagc, n_sampling=5)
    print_dagc(dagc, n_sampling=5, labels=["A", "B", "C", "D", "E"])


def test_make_prior_knowledge():
    pk = make_prior_knowledge(
        n_variables=5,
        exogenous_variables=[0],
        sink_variables=[1],
        paths=[[2, 3]],
        no_paths=[[3, 4]],
    )
    assert pk.shape[0] == 5 and pk.shape[1] == 5


def test_get_sink_variables():
    am = [
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
    ]
    sv = get_sink_variables(adjacency_matrix=am)
    assert len(sv) == 2


def test_get_exo_variables():
    am = [
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
    ]
    ev = get_exo_variables(adjacency_matrix=am)
    assert len(ev) == 2


def test_remove_effect():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0 * x1 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3]).T, columns=["x0", "x1", "x2", "x3"])

    remove_effect(X, remove_features=[0])

    # return_coefs
    remove_features = [0]
    remaining_features = [1, 2, 3]
    _, coefs = remove_effect(X, remove_features=remove_features, return_coefs=True)

    if list(coefs.keys()) != remaining_features:
        raise AssertionError

    for _, coef_list in coefs.items():
        if len(coef_list) != len(remove_features):
            raise AssertionError

def test_make_dot():
    # default
    am = [
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
    ]
    ev = make_dot(am, labels=["x0", "x1", "x2", "x3"])

    # default
    am = [
        [0, 1, 1, 0],
        [0, 0, np.nan, 0],
        [0, np.nan, 0, 0],
        [0, 1, 0, 0],
    ]
    ev = make_dot(am, labels=["x0", "x1", "x2", "x3"])

    am = [
        [0, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
    ]
    ev = make_dot(am, path=(3, 1), path_color="red", detect_cycle=True)

    # prediction
    ev = make_dot(am, prediction_feature_indices=[1, 2], prediction_coefs=[0.1, 0.1])
    ev = make_dot(
        am,
        prediction_feature_indices=[1, 2],
        prediction_target_label="Target",
        prediction_line_color="#0000FF",
    )
    ev = make_dot(
        am, prediction_feature_indices=[1, 2], prediction_feature_importance=[0.5, 0.2]
    )

    # invalid adjacency matrix
    am = [
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    try:
        ev = make_dot(am, labels=["x0", "x1", "x2", "x3"])
    except ValueError:
        pass
    else:
        raise AssertionError

    # invalid label size
    am = [
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
    ]
    try:
        ev = make_dot(am, labels=["x0", "x1", "x2"])
    except ValueError:
        pass
    else:
        raise AssertionError

    # invalid predict settings
    am = [
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
    ]
    try:
        ev = make_dot(am, prediction_feature_indices=[1, 2], prediction_coefs=[0.1])
    except ValueError:
        pass
    else:
        raise AssertionError

    try:
        ev = make_dot(
            am, prediction_feature_indices=[1, 2], prediction_feature_importance=[0.5]
        )
    except ValueError:
        pass
    else:
        raise AssertionError


def test_make_dot_highlight():
    # default
    am = [
        [0, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
    ]
    d = make_dot_highlight(am, 0)
    d = make_dot_highlight(am, 0, labels=["x0", "x1", "x2", "x3", "x4", "x5"])
    d = make_dot_highlight(am, 0, max_dsc=0, max_anc=None)
    d = make_dot_highlight(am, 0, max_dsc=0, max_anc=None, draw_others=False)
    d = make_dot_highlight(am, 0, detect_cycle=True)
    d = make_dot_highlight(am, 0, detect_cycle=True, lower_limit=0.1)
    d = make_dot_highlight(am, 0, lower_limit=0.001, cmap="cool", vmargin=3, hmargin=3)


def test_predict_adaptive_lasso():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0 * x1 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2, x3]).T, columns=["x0", "x1", "x2", "x3"])

    predict_adaptive_lasso(X.values, predictors=[0, 1, 2], target=3)


def test_find_all_paths():
    dag = [
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
    ]
    find_all_paths(np.array(dag), 1, 0)

    # min_causal_effect
    dag = [
        [0, 2, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
    ]
    find_all_paths(np.array(dag), 1, 0, min_causal_effect=1)

    # Invalid DAG: cycle
    dag = [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]
    try:
        find_all_paths(np.array(dag), 1, 0)
    except ValueError:
        pass
    else:
        raise AssertionError


def test_extract_ancestors():
    def get_coef():
        coef = random.random()
        return coef if coef >= 0.5 else coef - 1.0

    get_external_effect = lambda n: np.random.normal(0.0, 0.5, n) ** 3

    B = np.array(
        [
            [0.0, 0.0, get_coef(), 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, get_coef(), 0.0, 0.0],
            [0.0, 0.0, get_coef(), 0.0],
        ]
    )

    samples = 1000
    x1 = get_external_effect(samples)
    x2 = x1 * B[2, 1] + get_external_effect(samples)
    x0 = x2 * B[0, 2] + get_external_effect(samples)
    x3 = x2 * B[3, 2] + get_external_effect(samples)

    # x5, x6 is a latent variable.
    X = pd.DataFrame(np.array([x0, x1, x2, x3]).T, columns=["x0", "x1", "x2", "x3"])

    # Extract a set of ancestors of each variable
    extract_ancestors(X)


def corr_data(n, r):
    a = np.random.randn(n)
    e1 = np.random.randn(n)
    e2 = np.random.randn(n)
    if r < 0:
        r = -r
        x = -np.sqrt(r) * a - np.sqrt(1 - r) * e1
    else:
        x = np.sqrt(r) * a + np.sqrt(1 - r) * e1
    y = np.sqrt(r) * a + np.sqrt(1 - r) * e2
    return x, y


def test_f_correlation():
    # size < 1000
    x, y = corr_data(500, r=0.3)
    f_correlation(x, y)

    # size > 1000
    x, y = corr_data(1500, r=0.3)
    f_correlation(x, y)

    # len(x) != len(y)
    try:
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3, 4])
        f_correlation(x, y)
    except ValueError:
        pass
    else:
        raise AssertionError

def test_evaluate_model_fit():
    graph = np.array([
        [0, 0, 0],
        [0.5, 0, 0],
        [0.5, 0.5, 0],
    ])

    graph2 = np.array([
        [0, 0, 0],
        [0.5, 0, 0],
        [0.5, 0, 0],
    ])

    sample_size = 100
    X = np.zeros((sample_size, graph.shape[1]))
    X[:, 0] = np.random.normal(0, 1, size=sample_size)
    X[:, 1] = X @ graph[1, :]
    X[:, 1] += np.random.normal(0, np.std(X[:, 1]), size=sample_size)
    X[:, 2] = X @ graph[2, :]
    X[:, 2] += np.random.normal(0, np.std(X[:, 2]), size=sample_size)
    X = pd.DataFrame(X, columns=[f"x{i}" for i in range(graph.shape[1])])

    result = evaluate_model_fit(graph, X)
    result2 = evaluate_model_fit(graph2, X)
    assert result["LogLik"].values[0] < result2["LogLik"].values[0]

    # ordinal
    sample_size = 100
    X2 = np.zeros((sample_size, graph.shape[1]))
    X2[:, 0] = np.random.choice([0, 1], size=sample_size)
    X2[:, 1] = X2 @ graph[1, :]
    X2[:, 1] += np.random.normal(0, np.std(X2[:, 1]), size=sample_size)
    X2[:, 2] = X2 @ graph[2, :]
    X2[:, 2] += np.random.normal(0, np.std(X2[:, 2]), size=sample_size)
    X2 = pd.DataFrame(X2, columns=[f"x{i}" for i in range(graph.shape[1])])

    result = evaluate_model_fit(graph, X2, is_ordinal=[1, 0, 0])
    assert result["LogLik"].values[0] > 0

    # latent
    graph3 = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 1, 0],
    ])

    sample_size = 100
    X3 = np.zeros((sample_size, graph3.shape[1]))
    X3[:, 0] = np.random.normal(0, 1, size=sample_size)
    X3[:, 1] = X3 @ graph3[1, :]
    X3[:, 1] += np.random.normal(0, np.std(X3[:, 1]), size=sample_size)
    X3[:, 2] = X3 @ graph3[2, :]
    X3[:, 2] += np.random.normal(0, np.std(X3[:, 2]), size=sample_size)
    X3[:, 3] = X3 @ graph3[3, :]
    X3[:, 3] += np.random.normal(0, np.std(X3[:, 3]), size=sample_size)
    X3 = X3[:, 1:]
    X3 = pd.DataFrame(X3, columns=[f"x{i}" for i in range(X3.shape[1])])

    graph3 = np.array([
        [0, np.nan, 0],
        [np.nan, 0, 0],
        [1, 1, 0],
    ])

    result = evaluate_model_fit(graph3, X3)
    assert result["LogLik"].values[0] > 0

    # exception
    try:
        evaluate_model_fit([[0, 1]], X)
    except ValueError:
        pass
    else:
        raise AssertionError

    try:
        evaluate_model_fit(graph, X.iloc[:, :-1])
    except ValueError:
        pass
    else:
        raise AssertionError

    try:
        evaluate_model_fit(graph, X, is_ordinal=[0])
    except ValueError:
        pass
    else:
        raise AssertionError


def test_calculate_distance_from_root_nodes():
    # default
    adj = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0],
    ])
    result = calculate_distance_from_root_nodes(adj)
    assert result == {0: [0], 1: [1, 2], 2: [3]}

    # max_distance
    result = calculate_distance_from_root_nodes(adj, max_distance=1)
    assert result == {0: [0], 1: [1, 2]}

    # hidden common cause
    adj = np.array([
        [0, np.nan, 0, 0],
        [np.nan, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
    ])
    result = calculate_distance_from_root_nodes(adj)
    assert result == {0: [0, 1], 1: [2, 3]}

    # only hidden common cause
    adj = np.array([
        [0, np.nan],
        [np.nan, 0],
    ])
    result = calculate_distance_from_root_nodes(adj)
    assert result == {0: [0, 1]}

    # cyclic graph
    adj = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ])
    result = calculate_distance_from_root_nodes(adj)
    assert result == {0: [0], 1: [1], 2: [2], 3: [3]}

    # exception: not square
    adj = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
    ])
    try:
        calculate_distance_from_root_nodes(adj)
    except ValueError:
        pass
    else:
        raise AssertionError

    # exception: no root node
    adj = np.array([
        [0, 1],
        [1, 0],
    ])
    try:
        calculate_distance_from_root_nodes(adj)
    except ValueError:
        pass
    else:
        raise AssertionError


def test_calculate_total_effect():
    # default
    adj = np.array([
        [0, 0, 0],
        [0.5, 0, 0],
        [0, 0.5, 0],
    ])
    total_effect = calculate_total_effect(adj, 0, 2)
    assert np.isclose(total_effect, 0.5**2)

    # exception: continuous on the path
    is_con = [False, True, False]
    try:
        calculate_total_effect(adj, 0, 2, is_continuous=is_con)
    except ValueError:
        pass
    else:
        raise AssertionError

    # exception: not square
    adj = np.array([
        [0, 0, 0],
        [1, 0, 0],
    ])
    try:
        calculate_total_effect(adj, 0, 2)
    except ValueError:
        pass
    else:
        raise AssertionError


def test_get_common_edge_probabilities():
    class TestBootstrapResult():
        def __init__(self, adjacency_matrices):
            self._adjacency_matrices = adjacency_matrices

        @property
        def adjacency_matrices_(self):
            return self._adjacency_matrices

    results = []
    adjacency_matrices_1 = []
    adjacency_matrices_1.append(
        np.array([
            [0, 0, np.nan],
            [1, 0, 0],
            [np.nan, 1, 0]]))
    adjacency_matrices_1.append(
        np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0]]))

    adjacency_matrices_2 = []
    adjacency_matrices_2.append(
        np.array([
            [0, 0, np.nan],
            [0, 0, 0],
            [np.nan, 1, 0]]))
    adjacency_matrices_2.append(
        np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 0]]))

    results.append(TestBootstrapResult(adjacency_matrices_1))
    results.append(TestBootstrapResult(adjacency_matrices_2))
    get_common_edge_probabilities(results, mode="across")
    get_common_edge_probabilities(results, mode="per")

    # invalid adjacency matrix
    results = np.array([
        [0, 0, np.nan],
        [1, 0, 0],
        [np.nan, 1, 0]])
    try:
        get_common_edge_probabilities(results, mode="across")
    except ValueError:
        pass
    else:
        raise AssertionError

    # invalid adjacency matrix
    results = [adjacency_matrices_1, adjacency_matrices_2]
    try:
        get_common_edge_probabilities(results, mode="across")
    except AttributeError:
        pass
    else:
        raise AssertionError


def test_print_common_edge_directions():
    cep = np.array([[
        [0. , 0.5, 0.5, 0. , 0. , 0.],
        [0. , 0. , 0. , 0.5, 0. , 0.],
        [0. , 0.5, 0. , 0. , 0.5, 0.],
        [0. , 0. , 0. , 0. , 0. , 0.],
        [0. , 0.5, 0.5, 0.5, 0. , 0.],
        [0. , 0. , 0. , 0. , 0. , 0.]
    ], [
        [0. , 0. , 0. , 0. , 0. , 0.],
        [0. , 0. , 0.5, 0. , 0. , 0.],
        [0. , 0.5, 0. , 0.5, 0. , 0.],
        [0. , 0. , 0.5, 0. , 0. , 0.5],
        [0. , 0. , 0. , 0. , 0. , 0.],
        [0. , 0. , 0. , 0.5, 0. , 0.]]])
    print_common_edge_directions(cep)
    print_common_edge_directions(cep, labels=["A", "B", "C", "D", "E", "F"])
    print_common_edge_directions(cep, confidence_level=0.5)


def test_make_dot_for_nan_probability_matrix():
    # default
    am = [
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ]
    make_dot_for_nan_probability_matrix(am, labels=["x0", "x1", "x2", "x3"])

    # default
    am = [
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, np.nan],
        [0, 1, np.nan, 0],
    ]
    make_dot_for_nan_probability_matrix(am, labels=["x0", "x1", "x2", "x3"])

    am = [
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ]
    make_dot_for_nan_probability_matrix(am, labels=["x0", "x1", "x2", "x3"], path=(3, 1), path_color="red")

    # invalid adjacency matrix
    am = [
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    try:
        make_dot_for_nan_probability_matrix(am, labels=["x0", "x1", "x2", "x3"])
    except ValueError:
        pass
    else:
        raise AssertionError

    # invalid label size
    am = [
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
    ]
    try:
        make_dot_for_nan_probability_matrix(am, labels=["x0", "x1", "x2"])
    except ValueError:
        pass
    else:
        raise AssertionError
