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
