"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""

import numbers
import graphviz
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import semopy
import subprocess

from matplotlib import colors as mcolors
from matplotlib.colors import is_color_like
from sklearn import linear_model
from sklearn.linear_model import LassoLarsIC, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array, check_scalar

from ._rcd import extract_ancestors
from ._f_correlation import f_correlation
from ._visualize_nonlinear_causal_effect import visualize_nonlinear_causal_effect
from ._common_edge import get_common_edge_probabilities, print_common_edge_directions, make_dot_for_nan_probability_matrix

__all__ = [
    "print_causal_directions",
    "print_dagc",
    "make_prior_knowledge",
    "remove_effect",
    "make_dot",
    "make_dot_highlight",
    "predict_adaptive_lasso",
    "get_sink_variables",
    "get_exo_variables",
    "find_all_paths",
    "likelihood_i",
    "log_p_super_gaussian",
    "variance_i",
    "extract_ancestors",
    "f_correlation",
    "visualize_nonlinear_causal_effect",
    "evaluate_model_fit",
    "calculate_distance_from_root_nodes",
    "calculate_total_effect",
    "get_common_edge_probabilities",
    "print_common_edge_directions",
    "make_dot_for_nan_probability_matrix",
]


def print_causal_directions(cdc, n_sampling, labels=None):
    """Print causal directions of bootstrap result to stdout.

    Parameters
    ----------
    cdc : dict
        List of causal directions sorted by count in descending order.
        This can be set the value returned by ``BootstrapResult.get_causal_direction_counts()`` method.
    n_sampling : int
        Number of bootstrapping samples.
    labels : array-like, optional (default=None)
        List of feature lables.
        If set labels, the output feature name will be the specified label.
    """
    for i, (fr, to, co) in enumerate(zip(cdc["from"], cdc["to"], cdc["count"])):
        sign = "" if "sign" not in cdc else "(b>0)" if cdc["sign"][i] > 0 else "(b<0)"
        if labels:
            print(f"{labels[to]} <--- {labels[fr]} {sign} ({100*co/n_sampling:.1f}%)")
        else:
            print(f"x{to} <--- x{fr} {sign} ({100*co/n_sampling:.1f}%)")


def print_dagc(dagc, n_sampling, labels=None):
    """Print DAGs of bootstrap result to stdout.

    Parameters
    ----------
    dagc : dict
        List of directed acyclic graphs sorted by count in descending order.
        This can be set the value returned by ``BootstrapResult.get_directed_acyclic_graph_counts()`` method.
    n_sampling : int
        Number of bootstrapping samples.
    labels : array-like, optional (default=None)
        List of feature lables.
        If set labels, the output feature name will be the specified label.
    """
    for i, (dag, co) in enumerate(zip(dagc["dag"], dagc["count"])):
        print(f"DAG[{i}]: {100*co/n_sampling:.1f}%")
        for j, (fr, to) in enumerate(zip(dag["from"], dag["to"])):
            sign = (
                "" if "sign" not in dag else "(b>0)" if dag["sign"][j] > 0 else "(b<0)"
            )
            if labels:
                print("\t" + f"{labels[to]} <--- {labels[fr]} {sign}")
            else:
                print("\t" + f"x{to} <--- x{fr} {sign}")


def make_prior_knowledge(
    n_variables,
    exogenous_variables=None,
    sink_variables=None,
    paths=None,
    no_paths=None,
):
    """Make matrix of prior knowledge.

    Parameters
    ----------
    n_variables : int
        Number of variables.
    exogenous_variables : array-like, shape (index, ...), optional (default=None)
        List of exogenous variables(index).
        Prior knowledge is created with the specified variables as exogenous variables.
    sink_variables : array-like, shape (index, ...), optional (default=None)
        List of sink variables(index).
        Prior knowledge is created with the specified variables as sink variables.
    paths : array-like, shape ((index, index), ...), optional (default=None)
        List of variables(index) pairs with directed path.
        If ``(i, j)``, prior knowledge is created that xi has a directed path to xj.
    no_paths : array-like, shape ((index, index), ...), optional (default=None)
        List of variables(index) pairs without directed path.
        If ``(i, j)``, prior knowledge is created that xi does not have a directed path to xj.

    Returns
    -------
    prior_knowledge : array-like, shape (n_variables, n_variables)
        Return matrix of prior knowledge used for causal discovery.
    """
    prior_knowledge = np.full((n_variables, n_variables), -1)
    if no_paths:
        for no_path in no_paths:
            prior_knowledge[no_path[1], no_path[0]] = 0
    if paths:
        for path in paths:
            prior_knowledge[path[1], path[0]] = 1
    if sink_variables:
        for var in sink_variables:
            prior_knowledge[:, var] = 0
    if exogenous_variables:
        for var in exogenous_variables:
            prior_knowledge[var, :] = 0
    np.fill_diagonal(prior_knowledge, -1)
    return prior_knowledge


def get_sink_variables(adjacency_matrix):
    """The sink variables(index) in the adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : array-like, shape (n_variables, n_variables)
        Adjacency matrix, where n_variables is the number of variables.

    Returns
    -------
    sink_variables : array-like
        List of sink variables(index).
    """
    am = adjacency_matrix.copy()
    am = np.abs(am)
    np.fill_diagonal(am, 0)
    sink_vars = [i for i in range(am.shape[1]) if am[:, i].sum() == 0]
    return sink_vars


def get_exo_variables(adjacency_matrix):
    """The exogenous variables(index) in the adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : array-like, shape (n_variables, n_variables)
        Adjacency matrix, where n_variables is the number of variables.

    Returns
    -------
    exogenous_variables : array-like
        List of exogenous variables(index).
    """
    am = adjacency_matrix.copy()
    am = np.abs(am)
    np.fill_diagonal(am, 0)
    exo_vars = [i for i in range(am.shape[1]) if am[i, :].sum() == 0]
    return exo_vars


def remove_effect(X, remove_features, return_coefs=False):
    """Create a dataset that removes the effects of features by linear regression.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data, where ``n_samples`` is the number of samples
        and ``n_features`` is the number of features.
    remove_features : array-like, shape (n_removes,)
        List of features(index) to remove effects.
    return_coefs : bool, optional (default=False)
        Return regression coefficients or not.

    Returns
    -------
    X : array-like, shape (n_samples, n_features)
        Data after removing effects of ``remove_features``.
    coefs : dict, optional
        Coefficients estimated by linear regression.
        The keys are indices of remaining features and
        the values are lists of coefficients of removed features.
        The order of the coefficients in the list is the same
        as in remove_features. Only provided if return_coefs is True.
    """
    X = np.copy(check_array(X))
    coefs = {}
    features_ = [i for i in np.arange(X.shape[1]) if i not in remove_features]
    for feature in features_:
        reg = linear_model.LinearRegression()
        reg.fit(X[:, remove_features], X[:, feature])
        X[:, feature] = X[:, feature] - reg.predict(X[:, remove_features])
        coefs[feature] = reg.coef_
    if return_coefs:
        return X, coefs
    return X


def make_dot(
    adjacency_matrix,
    labels=None,
    lower_limit=0.01,
    prediction_feature_indices=None,
    prediction_target_label="Y(pred)",
    prediction_line_color="red",
    prediction_coefs=None,
    prediction_feature_importance=None,
    path=None,
    path_color=None,
    detect_cycle=False,
    ignore_shape=False,
):
    """Directed graph source code in the DOT language with specified adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : array-like with shape (n_features, n_features)
        Adjacency matrix to make graph, where ``n_features`` is the number of features.
    labels : array-like, optional (default=None)
        Label to use for graph features.
    lower_limit : float, optional (default=0.01)
        Threshold for drawing direction.
        If float, then directions with absolute values of coefficients less than ``lower_limit`` are excluded.
    prediction_feature_indices : array-like, optional (default=None)
        Indices to use as prediction features.
    prediction_target_label : string, optional (default='Y(pred)'))
        Label to use for target variable of prediction.
    prediction_line_color : string, optional (default='red')
        Line color to use for prediction's graph.
    prediction_coefs : array-like, optional (default=None)
        Coefficients to use for prediction's graph.
    prediction_feature_importance : array-like, optional (default=None)
        Feature importance to use for prediction's graph.
    path : tuple, optional (default=None)
        Path to highlight. Tuple of start index and end index.
    path_color : string, optional (default=None)
        Colors to highlight a path.
    detect_cycle : boolean, optional (default=False)
        Highlight simple cycles.
    ignore_shape : boolean, optional (default=False)
        Ignore checking the shape of adjaceny_matrix or not.

    Returns
    -------
    graph : graphviz.Digraph
        Directed graph source code in the DOT language.
        If order is unknown, draw a double-headed arrow.
    """
    # Check parameters
    B = check_array(np.nan_to_num(adjacency_matrix))
    if not ignore_shape and B.shape[0] != B.shape[1]:
        raise ValueError("'adjacency_matrix' is not square matrix.")
    if labels is not None:
        if B.shape[1] != len(labels):
            raise ValueError(
                "Length of 'labels' does not match length of 'adjacency_matrix'"
            )
    if prediction_feature_indices is not None:
        if prediction_coefs is not None and (
            len(prediction_feature_indices) != len(prediction_coefs)
        ):
            raise ValueError(
                "Length of 'prediction_coefs' does not match length of 'prediction_feature_indices'"
            )
        if prediction_feature_importance is not None and (
            len(prediction_feature_indices) != len(prediction_feature_importance)
        ):
            raise ValueError(
                "Length of 'prediction_feature_importance' does not match length of 'prediction_feature_indices'"
            )

    if path is not None:
        if not isinstance(path, tuple) or len(path) < 2:
            raise TypeError("'path' should be a tuple of node indices.")
        if path_color is not None and not is_color_like(path_color):
            raise ValueError("'path_color' should be an color name.")

        if path_color is None:
            path_color = "black"
        path_start = check_scalar(
            path[0], "path_start", int, min_val=0, max_val=B.shape[0]
        )
        path_end = check_scalar(path[1], "path_end", int, min_val=0, max_val=B.shape[0])

    if detect_cycle is not None:
        if not isinstance(detect_cycle, bool):
            raise TypeError("'detect_cycle' should be boolean.")

    idx = np.abs(B) > lower_limit

    nx_graph = None
    if path is not None or detect_cycle is True:
        nx_graph = nx.from_numpy_array(idx.astype(int).T, create_using=nx.DiGraph)

    # search path
    path_edges = None
    path_nodes = None
    if path is not None:
        path_edges = set()
        for path_ in nx.all_simple_paths(nx_graph, path[0], path[1]):
            path_edges |= set(zip(path_, path_[1:]))

        path_nodes = np.unique([list(edge) for edge in path_edges])
        if labels is None:
            path_nodes = [f"x{node}" for node in path_nodes]
        else:
            path_nodes = [labels[node] for node in path_nodes]

    # simple cycles
    cycle_edges = []
    if detect_cycle is True:
        for cycle in nx.simple_cycles(nx_graph):
            es = list(zip(cycle, np.roll(cycle, -1)))
            cycle_edges.append(es)
        cycle_edges = sum(cycle_edges, start=[])

    d = graphviz.Digraph(engine="dot")

    # nodes
    names = labels if labels else [f"x{i}" for i in range(len(B))]
    for name in names:
        kwargs = {}
        if path is not None:
            if name in path_nodes:
                kwargs["fontcolor"] = path_color
                kwargs["color"] = path_color
            if name == names[path_start] or name == names[path_end]:
                kwargs["peripheries"] = "2"
        d.node(name, **kwargs)

    # edges
    dirs = np.where(idx)
    for to, from_, coef in zip(dirs[0], dirs[1], B[idx]):
        kwargs = {}
        if path is not None and (from_, to) in path_edges:
            kwargs["penwidth"] = "2"
            kwargs["fontcolor"] = path_color
            kwargs["color"] = path_color
        if detect_cycle is True and (from_, to) in cycle_edges:
            kwargs["style"] = "dashed"
        d.edge(names[from_], names[to], label=f"{coef:.2f}", **kwargs)

    # integrate of prediction model
    if prediction_feature_indices is not None:
        d.node(
            prediction_target_label,
            color=prediction_line_color,
            fontcolor=prediction_line_color,
        )

        if prediction_coefs is not None:
            for from_, coef in zip(prediction_feature_indices, prediction_coefs):
                if np.abs(coef) > lower_limit:
                    d.edge(
                        names[from_],
                        prediction_target_label,
                        label=f"{coef:.2f}",
                        color=prediction_line_color,
                        fontcolor=prediction_line_color,
                        style="dashed",
                    )

        elif prediction_feature_importance is not None:
            for from_, imp in zip(
                prediction_feature_indices, prediction_feature_importance
            ):
                d.edge(
                    names[from_],
                    prediction_target_label,
                    label=f"({imp})",
                    color=prediction_line_color,
                    fontcolor=prediction_line_color,
                    style="dashed",
                )

        else:
            for from_ in prediction_feature_indices:
                d.edge(
                    names[from_],
                    prediction_target_label,
                    color=prediction_line_color,
                    style="dashed",
                )

    # If the value is nan, draw a double-headed arrow
    unk_order = np.where(np.isnan(np.tril(adjacency_matrix)))
    unk_order_set = set([val for item in unk_order for val in item])
    with d.subgraph() as s:
        s.attr(rank="same")
        for node in unk_order_set:
            s.node(names[node])
    for to, from_ in zip(unk_order[0], unk_order[1]):
        d.edge(names[from_], names[to], dir="both")

    return d


def make_dot_highlight(
    adjacency_matrix,
    node_index,
    labels=None,
    max_dsc=None,
    max_anc=None,
    lower_limit=None,
    cmap=None,
    draw_others=True,
    detect_cycle=False,
    path=None,
    path_color=None,
    vmargin=2,
    hmargin=2,
):
    """Hierachical make_dot.

    Parameters
    ----------
    adjacency_matrix : array-like with shape (n_features, n_features)
        Adjacency matrix to make graph, where ``n_features`` is the number of features.
    node_index : int
        Index of target node.
    labels : array-like, optional (default=None)
        Label to use for graph features.
    max_dsc : int, optional (default=None)
        Number of hierarchies on the descendant side to display.
    max_asc : int, optional (default=None)
        Number of hierarchies on the ancestral side to display.
    lower_limit : float, optional (default=0.01)
        Threshold for drawing direction.
        If float, then directions with absolute values of coefficients less than ``lower_limit`` are excluded.
    cmap : colormap, optional (default=None)
        Gradient color to represent hierarchy.
    draw_others : boolean, optional (default=True)
        Whether to draw others cluster.
    detect_cycle : boolean, optional (default=False)
        Highlight simple cycles.
    path : tuple, optional (default=None)
        Path to highlight. Tuple of start index and end index.
    path_color : string, optional (default=None)
        Colors to highlight a path.
    vmargin : float, optional (default=2)
        Vertical margin between nodes.
    hmargin : float, optional (default=2)
        Horizontal margin between nodes.

    Returns
    -------
    graph : graphviz.Digraph
        Directed graph source code in the DOT language.
        If order is unknown, draw a double-headed arrow.
    """

    # margins for clusters
    cls_margin_l = 1.5
    cls_margin_r = 0.6
    cls_margin_t = 0.6
    cls_margin_b = 0.4
    cls_space_target_other = 0.2

    # check arguments
    adj = check_array(
        adjacency_matrix,
        ensure_2d=True,
        ensure_min_samples=2,
        ensure_min_features=2,
        copy=True,
    )
    if adj.shape[0] != adj.shape[1]:
        raise ValueError

    node_index = check_scalar(
        node_index,
        "node_index",
        int,
        min_val=0,
        max_val=adj.shape[0],
        include_boundaries="left",
    )

    if labels is None:
        node_names = [f"x{i}" for i in range(adj.shape[1])]
    else:
        node_names = check_array(labels, ensure_2d=False, dtype=str)
        if len(node_names) != adj.shape[1]:
            raise ValueError

    if max_dsc is not None:
        max_dsc = check_scalar(max_dsc, "max_dsc", int, min_val=0)

    if max_anc is not None:
        max_anc = check_scalar(max_anc, "max_anc", int, min_val=0)

    if lower_limit is not None:
        lower_limit = check_scalar(lower_limit, "lower_limit", (int, float))

    draw_others = check_scalar(draw_others, "draw_others", bool)
    detect_cycle = check_scalar(detect_cycle, "detect_cycle", bool)

    path_start = None
    path_end = None
    if path is not None:
        if not isinstance(path, tuple) or len(path) < 2:
            raise TypeError("'path' should be a tuple of node indices.")
        if path_color is not None and not is_color_like(path_color):
            raise ValueError("'path_color' should be an color name.")
        if path_color is None:
            path_color = "black"
        path_start = check_scalar(
            path[0], "start", int, min_val=0, max_val=adj.shape[0]
        )
        path_end = check_scalar(path[1], "end", int, min_val=0, max_val=adj.shape[0])

    vmargin = check_scalar(vmargin, "vmargin", (float, int), min_val=1)
    hmargin = check_scalar(hmargin, "hmargin", (float, int), min_val=1)

    # apply lower_limit
    if lower_limit is not None:
        adj[abs(adj) < lower_limit] = 0

    # analyze network
    G = nx.from_numpy_array(~np.isclose(adj.T, 0), create_using=nx.DiGraph)

    dsc_path_length = nx.single_source_shortest_path_length(
        G, node_index, cutoff=max_dsc
    )
    del dsc_path_length[node_index]

    anc_path_length = nx.single_source_shortest_path_length(
        G.reverse(), node_index, cutoff=max_anc
    )
    del anc_path_length[node_index]

    # node indices
    dsc_indices = set(dsc_path_length.keys())
    anc_indices = set(anc_path_length.keys())
    isolate_indices = set(nx.isolates(G))
    other_indices = (
        set(np.arange(adj.shape[0]))
        - dsc_indices
        - anc_indices
        - isolate_indices
        - set([node_index])
    )

    # clusters (distance -> list of nodes)
    clusters = {}

    clusters[0] = [node_index]

    clusters[None] = sorted(other_indices) + sorted(isolate_indices)

    for node in dsc_indices:
        d = dsc_path_length[node]
        if d not in clusters.keys():
            clusters[d] = []
        clusters[d].append(node)

    for node in anc_indices:
        d = -anc_path_length[node]
        if d not in clusters.keys():
            clusters[d] = []
        clusters[d].append(node)

    # search path
    path_edges = None
    path_nodes = None
    if path is not None:
        path_edges = set()
        for path_ in nx.all_simple_paths(G, path_start, path_end):
            path_edges |= set(zip(path_, path_[1:]))

        path_nodes = np.unique([list(edge) for edge in path_edges])
        path_nodes = [node_names[node] for node in path_nodes]

    # create graphviz graph
    graph_attr = {"rankdir": "TB", "splines": "true"}
    graph = graphviz.Digraph("graph", engine="neato", graph_attr=graph_attr)

    # colors of clusters
    ds = sorted([d for d in clusters.keys() if d is not None])
    if cmap is None:
        cluster_colors = {d: "black" for d in ds}
    else:
        cmap = plt.get_cmap(cmap)
        cs = np.linspace(0, 1, len(ds) + 1)
        cluster_colors = dict(zip(ds, cmap(cs)))
    cluster_colors[None] = cluster_colors[0]

    # distance -> y_position
    pos_y_map = {0: 0, None: 0}

    # distance from target to descendants -> pos_y
    dsc_dists = sorted([d for d in clusters.keys() if d is not None and 0 < d])
    for i, dist in enumerate(dsc_dists):
        pos_y_map[dist] = -(i + 1) * vmargin

    # distance from target to ascendants -> pos_y
    anc_dists = sorted([d for d in clusters.keys() if d is not None and d < 0], key=abs)
    for i, dist in enumerate(anc_dists):
        pos_y_map[dist] = (i + 1) * vmargin

    get_cluster_name = lambda dist: f"cluster{dist}"
    get_node_id = lambda cl, node: f"{cls_name}-{node}"

    node_ids = {node: [] for node in np.arange(adj.shape[0])}

    # add subgraphs and nodes
    for dist, nodes in clusters.items():
        # skip drawing the cluster
        if dist is None and draw_others is False:
            continue

        if len(nodes) == 0:
            continue

        # attributes for the cluster and nodes
        if dist == 0:
            cls_label = "target"
            offset_x = 0
        elif dist is None:
            cls_label = "others"
            offset_x = cls_margin_r + cls_margin_l + cls_space_target_other
            nodes = sorted(
                nodes, key=lambda x: (x in isolate_indices, x in other_indices, x)
            )
        else:
            cls_label = (
                f"ancestor {abs(dist)}" if dist < 0 else f"descendant {abs(dist)}"
            )
            offset_x = 0
            nodes = sorted(nodes)
        pos_y = pos_y_map[dist]
        color = mcolors.to_hex(cluster_colors[dist], keep_alpha=True)
        cls_name = get_cluster_name(dist)

        graph_attr = {
            "style": "dashed",
            "label": cls_label,
            "labeljust": "l",
            "color": color,
        }
        with graph.subgraph(name=cls_name, graph_attr=graph_attr) as c:
            # add nodes
            for i, node in enumerate(nodes):
                node_id = get_node_id(cls_name, node)
                pos_x = hmargin * i + offset_x
                kwargs = {}
                if path is not None:
                    if node in path_nodes:
                        kwargs["fontcolor"] = path_color
                        kwargs["color"] = path_color
                    if node == path_start or node == path_end:
                        kwargs["peripheries"] = "2"
                c.node(
                    node_id,
                    node_names[node],
                    pos=f"{pos_x},{pos_y}!",
                    color=color,
                    **kwargs,
                )
                node_ids[node].append((node_id, dist))

            # dummy at the upper left
            x = -cls_margin_l + offset_x
            y = pos_y + cls_margin_t
            c.node(
                f"{cls_name}-dummy_upper",
                "",
                pos=f"{x},{y}!",
                style="invis",
                fixedsize="true",
                width="0",
                height="0",
            )

            # dummy at the lower right
            x = (len(nodes) - 1) * hmargin + cls_margin_r + offset_x
            y = pos_y - cls_margin_b
            c.node(
                f"{cls_name}-dummy_lower",
                "",
                pos=f"{x},{y}!",
                style="invis",
                fixedsize="true",
                width="0",
                height="0",
            )

    # cycles
    cycle_edges = []
    if detect_cycle is True:
        for cycle in nx.simple_cycles(G):
            es = list(zip(cycle, np.roll(cycle, -1)))
            cycle_edges.append(es)
        cycle_edges = sum(cycle_edges, start=[])

    # indices of clusters
    other_nodes = isolate_indices | other_indices
    valid_nodes = other_nodes | dsc_indices | anc_indices | set([node_index])

    # add edges
    edges = np.argwhere(~np.isclose(adj, 0))
    for h, t in edges:
        if draw_others is False and (t in other_nodes or h in other_nodes):
            continue

        if t not in valid_nodes or h not in valid_nodes:
            continue

        for tail_id, dist_t in node_ids[t]:
            for head_id, dist_h in node_ids[h]:
                color = cluster_colors[dist_h]
                color = mcolors.to_hex(color, keep_alpha=True)

                kwargs = {"color": color}
                if path is not None and (t, h) in path_edges:
                    kwargs["penwidth"] = "2"
                    kwargs["fontcolor"] = path_color
                    kwargs["color"] = path_color
                if detect_cycle is True and (t, h) in cycle_edges:
                    kwargs["style"] = "dashed"

                graph.edge(tail_id, head_id, label=f"{adj[h, t]:.2f}", **kwargs)

    return graph


def predict_adaptive_lasso(X, predictors, target, gamma=1.0):
    """Predict with Adaptive Lasso.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples
        and n_features is the number of features.
    predictors : array-like, shape (n_predictors)
        Indices of predictor variable.
    target : int
        Index of target variable.

    Returns
    -------
    coef : array-like, shape (n_features)
        Coefficients of predictor variable.
    """
    # Standardize X
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Pruning with Adaptive Lasso
    lr = LinearRegression()
    lr.fit(X_std[:, predictors], X_std[:, target])
    weight = np.power(np.abs(lr.coef_), gamma)
    reg = LassoLarsIC(criterion="bic")
    reg.fit(X_std[:, predictors] * weight, X_std[:, target])
    pruned_idx = np.abs(reg.coef_ * weight) > 0.0

    # Calculate coefficients of the original scale
    coef = np.zeros(reg.coef_.shape)
    if pruned_idx.sum() > 0:
        lr = LinearRegression()
        pred = np.array(predictors)
        lr.fit(X[:, pred[pruned_idx]], X[:, target])
        coef[pruned_idx] = lr.coef_

    return coef


def find_all_paths(dag, from_index, to_index, min_causal_effect=0.0):
    """Find all paths from point to point in DAG.

    Parameters
    ----------
    dag : array-like, shape (n_features, n_features)
        The adjacency matrix to fine all paths, where n_features is the number of features.
    from_index : int
        Index of the variable at the start of the path.
    to_index : int
        Index of the variable at the end of the path.
    min_causal_effect : float, optional (default=0.0)
        Threshold for detecting causal direction.
        Causal directions with absolute values of causal effects less than ``min_causal_effect`` are excluded.

    Returns
    -------
    paths : array-like, shape (n_paths)
        List of found path, where n_paths is the number of paths.
    effects : array-like, shape (n_paths)
        List of causal effect, where n_paths is the number of paths.
    """
    # Extract all edges
    edges = np.array(np.where(np.abs(np.nan_to_num(dag)) > min_causal_effect)).T

    # Aggregate edges by start point
    to_indices = []
    for i in range(dag.shape[0]):
        adj_list = edges[edges[:, 1] == i][:, 0].tolist()
        if len(adj_list) != 0:
            to_indices.append(adj_list)
        else:
            to_indices.append([])

    # DFS
    paths = []
    stack = [from_index]
    stack_to_indice = [to_indices[from_index]]
    while stack:
        if len(stack) > dag.shape[0]:
            raise ValueError(
                "Unable to find the path because a cyclic graph has been specified."
            )

        cur_index = stack[-1]
        to_indice = stack_to_indice[-1]

        if cur_index == to_index:
            paths.append(stack.copy())
            stack.pop()
            stack_to_indice.pop()
        else:
            if len(to_indice) > 0:
                next_index = to_indice.pop(0)
                stack.append(next_index)
                stack_to_indice.append(to_indices[next_index].copy())
            else:
                stack.pop()
                stack_to_indice.pop()

    # Calculate the causal effect for each path
    effects = []
    for p in paths:
        coefs = [dag[p[i + 1], p[i]] for i in range(len(p) - 1)]
        effects.append(np.cumprod(coefs)[-1])

    return paths, effects


def likelihood_i(x, i, b_i, bi_0):
    """Compute local log-likelihood of component i.

    Parameters
    ----------
    x : array-like, shape (n_features, n_samples)
        Data, where ``n_samples`` is the number of samples
        and ``n_features`` is the number of features.
    i : array-like
        Variable index.
    b_i : array-like
        The i^th column of adjacency matrix, B[i].
    bi_0 : float
        Constant value for the i^th variable.

    Return
    -------
    ll : float
        Local log-likelihood of component i.
    """

    sample_size = x.shape[1]  # number of data points
    var_i = variance_i(x, i, b_i)
    #
    ll = 0.0
    ll += np.sum(log_p_super_gaussian((x[i] - np.dot(b_i, x) - bi_0) / np.sqrt(var_i)))

    ll -= sample_size * np.log(np.sqrt(var_i))

    return ll


def log_p_super_gaussian(s):
    """Compute density function of the normalized independent components.

    Parameters
    ----------
    s : array-like, shape (1, n_samples)
        Data, where ``n_samples`` is the number of samples.

    Return
    -------
    x : float
        Density function of the normalized independent components,
        whose disturbances are super-Gaussian.
    """
    const = -0.35  # normalising constant
    return -np.sqrt(2.0) * np.absolute(s) + const


def variance_i(X, i, b_i):
    """Compute empirical variance of component i.

    Parameters
    ----------
    x : array-like, shape (n_features, n_samples)
        Data, where ``n_samples`` is the number of samples
        and ``n_features`` is the number of features.
    i : array-like
        Variable index.
    b_i : array-like
        The i^th column of adjacency matrix, B[i].

    Return
    -------
    variance : float
        Empirical variance of component i.
    """
    # T = X.shape[1]  # sample size
    estimated_disturbance = X[i] - np.dot(b_i, X)
    # variance = np.sum(estimated_disturbance ** 2) / T  # JMLR paper assumes zero mean
    variance = np.var(estimated_disturbance)  # stable version, even not zero mean.

    return variance


def evaluate_model_fit(adjacency_matrix, X, is_ordinal=None):
    """evaluate the given adjacency matrix and return fit indices

    Parameters
    ----------
    adjacency_matrix : array-like, shape (n_features, n_features)
        Adjacency matrix representing a causal graph.
        The i-th column and row correspond to the i-th column of X.
    X : array-like, shape (n_samples, n_features)
        Training data.
    is_ordinal : array-like, shape (n_features,)
        Binary list. The i-th element represents that the i-th column of X is ordinal or not.
        0 means not ordinal, otherwise ordinal.

    Return
    ------
    fit_indices : pandas.DataFrame
        Fit indices. This API uses semopy's calc_stats(). See semopy's reference for details.
    """

    # check inputs
    adj = check_array(adjacency_matrix, force_all_finite="allow-nan")
    if adj.ndim != 2 or (adj.shape[0] != adj.shape[1]):
        raise ValueError("adj must be an square matrix.")

    X = check_array(X)
    if X.shape[1] != adj.shape[1]:
        raise ValueError("X.shape[1] and adj.shape[1] must be the same.")

    if is_ordinal is None:
        is_ordinal = np.zeros(X.shape[1])
    else:
        is_ordinal = check_array(is_ordinal, ensure_2d=False).flatten()
    if is_ordinal.shape[0] != adj.shape[1]:
        raise ValueError("is_ordinal.shape[0] and adj.shape[1] must be the same.")

    # build desc
    desc = ""
    eta_names = []

    for i, row in enumerate(adj):
        # exogenous
        if np.sum(np.isnan(row)) == 0 and np.sum(np.isclose(row, 0)) == row.shape[0]:
            continue

        desc += f"x{i:d} ~ "

        for j, elem in enumerate(row):
            if np.isnan(elem):
                eta_name = f"eta_{i}_{j}" if i < j else f"eta_{j}_{i}"
                desc += f"{eta_name} + "
                if eta_name not in eta_names:
                    eta_names.append(eta_name)
            elif not np.isclose(elem, 0):
                desc += f"{elem:f} * x{j:d} + "
        desc = desc[: -len(" * ")] + "\n"

    if len(eta_names) > 0:
        desc += "DEFINE(latent) " + " ".join(eta_names) + "\n"

    if sum(is_ordinal) > 0:
        indices = np.argwhere(is_ordinal).flatten()

        desc += "DEFINE(ordinal)"
        for i in indices:
            desc += f" x{i}"
        desc += "\n"

    columns = [f"x{i:d}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=columns)

    m = semopy.Model(desc)
    m.fit(X)

    stats = semopy.calc_stats(m)

    return stats


def calculate_distance_from_root_nodes(adjacency_matrix, max_distance=None):
    """Calculate shortest distances from root nodes.

    Parameters
    ----------
    adjacency_matrices : array-like
        The adjacency matrix.
    max_distance : int or None, optional (default=None)
        The maximum distance to return nodes from root nodes.

    Returns
    -------
    shortest_distances : dict
        The dictionary has the following format::

        {'distance_from_root_node': [index_of_variables]}
    """
    # check inputs
    adjacency_matrix = check_array(
        adjacency_matrix,
        ensure_min_samples=2,
        ensure_min_features=2,
        force_all_finite="allow-nan",
    )
    if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError("adjacency_matrix must be an square matrix.")

    if max_distance is not None:
        max_distance = check_scalar(max_distance, "max_distance", int, min_val=1)

    # delete hidden common causes
    adjacency_matrix = np.nan_to_num(adjacency_matrix)

    # find root nodes
    root_indices = np.argwhere(
        np.isclose(np.sum(adjacency_matrix, axis=1), 0) == True
    ).flatten()
    if len(root_indices) == 0:
        raise ValueError("adjacency_matrix has no root nodes.")

    G = nx.from_numpy_array(adjacency_matrix.T, create_using=nx.DiGraph)

    # distances from root nodes
    dist_df = {}
    for index in root_indices:
        dist = nx.shortest_path_length(G, source=index)
        dist_df[index] = dist
    dist_df = pd.DataFrame(dist_df)

    dist_df = dist_df.fillna(np.iinfo(int).max)
    dist_df = dist_df.min(axis=1)
    if max_distance is not None:
        dist_df = dist_df.iloc[dist_df.values <= max_distance]
    dist_df = dist_df.sort_index()
    dist_df = dist_df.astype(int)

    result = {i: [] for i in pd.unique(dist_df)}
    for name, dist in dist_df.items():
        result[dist].append(name)

    return result


def calculate_total_effect(adjacency_matrix, from_index, to_index, is_continuous=None):
    """Calculate total effect.

    Parameters
    ----------
    adjacency_matrix : array_like
        The adjacency matrix.
    from_index : int
        The index of the cause variable.
    to_index : int
        The index of the effect variable.
    is_continuous : list
        The list of boolean. is_continuous indicates whether each variable
        is continuous or discrete.

    Returns
    -------
    total_effect : float
    """

    # check inputs
    adjacency_matrix = check_array(
        adjacency_matrix,
        ensure_min_samples=2,
        ensure_min_features=2,
        force_all_finite="allow-nan",
    )
    if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError(
            "adjacency_matrix must be an square matrix.", adjacency_matrix.shape
        )

    from_index = check_scalar(
        from_index,
        "from_index",
        (numbers.Integral, np.integer),
        min_val=0,
        max_val=len(adjacency_matrix) - 1,
    )

    to_index = check_scalar(
        to_index,
        "to_index",
        (numbers.Integral, np.integer),
        min_val=0,
        max_val=len(adjacency_matrix) - 1,
    )

    if from_index == to_index:
        raise ValueError("from_index and to_index mustn't be the same.")

    if is_continuous is None:
        is_continuous = [True for _ in range(len(adjacency_matrix))]
    else:
        is_continuous = check_array(
            is_continuous, ensure_2d=False, ensure_min_samples=len(adjacency_matrix)
        )

    # find all paths
    path_list, effects = find_all_paths(adjacency_matrix, from_index, to_index)

    # check all nodes on the path are continuous
    for path in path_list:
        for node in path[1:]:
            if not is_continuous[node]:
                raise ValueError("Variables on the path must be continuous variables.")

    total_effect = sum(effects)

    return total_effect

def get_cuda_version():
    try:
        nvcc_version = subprocess.check_output(["nvcc", "--version"]).decode('utf-8')
        print("CUDA Version found:\n", nvcc_version)
        return True
    except Exception as e:
        print("CUDA not found or nvcc not in PATH:", e)
        return False
