"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""
import graphviz
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import is_color_like
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LassoLarsIC, LinearRegression
from sklearn.utils import check_array, check_scalar
import networkx as nx

import igraph as ig
from scipy.special import expit as sigmoid
import random

from ._rcd import extract_ancestors


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
    "simulate_dag",
    "simulate_parameter",
    "simulate_linear_sem",
    "simulate_linear_mixed_sem",
    "is_dag",
    "count_accuracy",
    "set_random_seed",
    "likelihood_i",
    "log_p_super_gaussian",
    "variance_i",
    "extract_ancestors",
]


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def simulate_linear_sem(adjacency_matrix, n_samples, sem_type, noise_scale=1.0):
    """Simulate samples from linear SEM with specified type of noise.

    Parameters
    ----------
    adjacency_matrix : array-like, shape (n_features, n_features)
        Weighted adjacency matrix of DAG, where ``n_features``
        is the number of variables.
    n_samples : int
        Number of samples. n_samples=inf mimics population risk.
    sem_type : str
        SEM type. gauss, exp, gumbel, logistic, poisson.
    noise_scale : float
        scale parameter of additive noise.

    Returns
    -------
    X : array-like, shape (n_samples, n_features)
        Data generated from linear SEM with specified type of noise,
        where ``n_features`` is the number of variables.
    """

    def _simulate_single_equation(X, w):
        """Simulate samples from a single equation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features_parents)
            Data of parents for a specified variable, where
            n_features_parents is the number of parents.
        w : array-like, shape (1, n_features_parents)
            Weights of parents.

        Returns
        -------
        x : array-like, shape (n_samples, 1)
            Data for the specified variable.
        """
        if sem_type == "gauss":
            z = np.random.normal(scale=noise_scale, size=n_samples)
            x = X @ w + z
        elif sem_type == "exp":
            z = np.random.exponential(scale=noise_scale, size=n_samples)
            x = X @ w + z
        elif sem_type == "gumbel":
            z = np.random.gumbel(scale=noise_scale, size=n_samples)
            x = X @ w + z
        elif sem_type == "logistic":
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == "poisson":
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        elif sem_type == "subGaussian":
            z = np.random.normal(scale=noise_scale, size=n_samples)
            q = 0.5 + 0.3 * np.random.rand(1)  # sub-Gaussian
            z = np.sign(z) * pow(np.abs(z), q)
            z = z - np.mean(z)
            z = z / np.std(z)
            x = X @ w + z
        elif sem_type == "supGaussian":
            z = np.random.normal(scale=noise_scale, size=n_samples)
            q = 1.2 + 0.8 * np.random.rand(1)  # super-Gaussian
            z = np.sign(z) * pow(np.abs(z), q)
            z = z - np.mean(z)
            z = z / np.std(z)
            x = X @ w + z
        elif sem_type == "nonGaussian":
            z = np.random.normal(scale=noise_scale, size=n_samples)
            qq = -1
            if qq == 1:
                q = 0.5 + 0.3 * np.random.rand(1)  # sub-Gaussian
            else:
                q = 1.2 + 0.8 * np.random.rand(1)  # super-Gaussian
            z = np.sign(z) * pow(np.abs(z), q)
            z = z - np.mean(z)
            z = z / np.std(z)
            x = X @ w + z
        elif sem_type == "uniform":
            z = np.random.uniform(0, 1, n_samples)
            z = z - np.mean(z)
            z = z / np.std(z)
            x = X @ w + z
        elif sem_type == "gamma":
            z = np.random.gamma(2, 2, n_samples)
            z = z - np.mean(z)
            z = z / np.std(z)
            x = X @ w + z
        elif sem_type == "laplace":
            z = np.random.laplace(0, scale=noise_scale, size=n_samples)
            x = X @ w + z
        else:
            raise ValueError("unknown sem type")
        return x

    n_features = adjacency_matrix.shape[0]
    if np.isinf(n_samples):
        if sem_type == "gauss":
            # make 1/n_features X'X = true cov
            X = (
                np.sqrt(n_features)
                * noise_scale
                * np.linalg.pinv(np.eye(n_features) - adjacency_matrix)
            )
            return X
        else:
            raise ValueError("population risk not available")
    X = np.zeros([n_samples, n_features])

    G = ig.Graph.Weighted_Adjacency(adjacency_matrix.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == n_features

    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], adjacency_matrix[parents, j])
    return X


def simulate_linear_mixed_sem(
    adjacency_matrix, n_samples, sem_type, dis_con, noise_scale=None
):
    """Simulate mixed samples from linear SEM with specified type of noise.

    Parameters
    ----------
    adjacency_matrix : array-like, shape (n_features, n_features)
        Weighted adjacency matrix of DAG, where ``n_features``
        is the number of variables.
    n_samples : int
        Number of samples. n_samples=inf mimics population risk.
    sem_type : str
        SEM type. gauss, mixed_random_i_dis.
    dis_con : array-like, shape (1, n_features)
        Indicator of discrete/continuous variables, where "1"
        indicates a continuous variable, while "0" a discrete
        variable.
    noise_scale : float
        scale parameter of additive noise.

    Returns
    -------
    X : array-like, shape (n_samples, n_features)
        Data generated from linear SEM with specified type of noise,
        where ``n_features`` is the number of variables.
    """

    def _simulate_single_equation(X, w, scale, dis_con_j):
        """Simulate samples from a single equation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features_parents)
            Data of parents for a specified variable, where
            n_features_parents is the number of parents.
        w : array-like, shape (1, n_features_parents)
            Weights of parents.
        scale : scale parameter of additive noise.
        dis_con_j : indicator of the j^th variable.

        Returns
        -------
        x : array-like, shape (n_samples, 1)
                    Data for the specified variable.
        """
        if sem_type == "gauss":
            z = np.random.normal(scale=scale, size=n_samples)
            x = X @ w + z
        elif sem_type == "mixed_random_i_dis":
            # randomly generated with fixed number of discrete variables.
            if dis_con_j:  # 1:continuous;   0:discrete
                z = np.random.laplace(0, scale=scale, size=n_samples)
                x = X @ w + z
            else:
                x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        else:
            raise ValueError("unknown sem type")
        return x

    n_features = adjacency_matrix.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(n_features)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(n_features)
    else:
        if len(noise_scale) != n_features:
            raise ValueError("noise scale must be a scalar or has length n_features")
        scale_vec = noise_scale
    if not is_dag(adjacency_matrix):
        raise ValueError("adjacency_matrix must be a DAG")
    if np.isinf(n_samples):  # population risk for linear gauss SEM
        if sem_type == "gauss":
            # make 1/n_features X'X = true cov
            X = (
                np.sqrt(n_features)
                * np.diag(scale_vec)
                @ np.linalg.inv(np.eye(n_features) - adjacency_matrix)
            )
            return X
        else:
            raise ValueError("population risk not available")
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(adjacency_matrix.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == n_features
    X = np.zeros([n_samples, n_features])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        # X[:, j] = _simulate_single_equation(X[:, parents], adjacency_matrix[parents, j], scale_vec[j])
        X[:, j] = _simulate_single_equation(
            X[:, parents], adjacency_matrix[parents, j], scale_vec[j], dis_con[0, j]
        )
    return X


def is_dag(W):
    """Check if W is a dag or not.

    Parameters
    ----------
    W : array-like, shape (n_features, n_features)
        Binary adjacency matrix of DAG, where ``n_features``
        is the number of features.

    Returns
    -------
     G: boolean
        Returns true or false.

    """
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def count_accuracy(W_true, W, W_und=None):
    """Compute recalls and precisions for W, or optionally for CPDAG = W + W_und.

    Parameters
    ----------
    W_true : array-like, shape (n_features, n_features)
        Ground truth graph, where ``n_features`` is
        the number of features.
    W : array-like, shape (n_features, n_features)
        Predicted graph.
    W_und : array-like, shape (n_features, n_features)
        Predicted undirected edges in CPDAG, asymmetric.

    Returns
    -------
    recall : float
        (true positive) / (true positive + false negative).
    precision : float
        (true positive) / (true positive + false positive).
    """
    # convert to binary adjacency matrix
    B_true = W_true != 0
    B = W != 0
    B_und = None if W_und is None else (W_und != 0)
    # linear index of nonzeros
    pred_und = None
    if B_und is not None:
        pred_und = np.flatnonzero(B_und)
    pred = np.flatnonzero(B)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    if B_und is not None:
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    if B_und is not None:
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    # extra = np.setdiff1d(pred, cond, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    if B_und is not None:
        pred_size += len(pred_und)
    # fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    # fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)

    recall = tpr
    precision = float(len(true_pos)) / max(pred_size, 1)

    return recall, precision


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.

    Parameters
    ----------
    B : array-like, shape (n_features, n_features)
        Binary adjacency matrix of DAG, where ``n_features``
        is the number of features.
    w_ranges : tuple
        Disjoint weight ranges.

    Returns
    -------
    adjacency_matrix : array-like, shape (n_features, n_features)
        Weighted adj matrix of DAG, where ``n_features``
        is the number of features.
    """

    adjacency_matrix = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        adjacency_matrix += B * (S == i) * U
    return adjacency_matrix


def simulate_dag(n_features, n_edges, graph_type):
    """Simulate random DAG with some expected number of edges.

    Parameters
    ----------
    n_features : int
        Number of features.
    n_edges : int
        Expected number of edges.
    graph_type : str
        ER, SF.

    Returns
    -------
    B : array-like, shape (n_features, n_features)
        binary adjacency matrix of DAG.
    """

    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == "ER":
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=n_features, m=n_edges)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == "SF":
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(
            n=n_features, m=int(round(n_edges / n_features)), directed=True
        )
        B = _graph_to_adjmat(G)
    elif graph_type == "BP":
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * n_features)
        G = ig.Graph.Random_Bipartite(
            top, n_features - top, m=n_edges, directed=True, neimode=ig.OUT
        )
        B = _graph_to_adjmat(G)
    else:
        raise ValueError("unknown graph type")
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


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


def remove_effect(X, remove_features):
    """Create a dataset that removes the effects of features by linear regression.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data, where ``n_samples`` is the number of samples
        and ``n_features`` is the number of features.
    remove_features : array-like
        List of features(index) to remove effects.

    Returns
    -------
    X : array-like, shape (n_samples, n_features)
        Data after removing effects of ``remove_features``.
    """
    X = np.copy(check_array(X))
    features_ = [i for i in np.arange(X.shape[1]) if i not in remove_features]
    for feature in features_:
        reg = linear_model.LinearRegression()
        reg.fit(X[:, remove_features], X[:, feature])
        X[:, feature] = X[:, feature] - reg.predict(X[:, remove_features])
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
        path_start = check_scalar(path[0], "path_start", int, min_val=0, max_val=B.shape[0])
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
    hmargin=2
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
    adj = check_array(adjacency_matrix, ensure_2d=True, ensure_min_samples=2, 
            ensure_min_features=2, copy=True)
    if adj.shape[0] != adj.shape[1]:
        raise ValueError
    
    node_index = check_scalar(node_index, "node_index", int, min_val=0,
            max_val=adj.shape[0], include_boundaries="left")
    
    if labels is None:
        node_names = [f"x{i}" for i in range(adj.shape[1])]
    else:
        node_names = check_array(node_names, ensure_2d=False, dtype=str)
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
        path_start = check_scalar(path[0], "start", int, min_val=0, max_val=adj.shape[0])
        path_end = check_scalar(path[1], "end", int, min_val=0, max_val=adj.shape[0])
   
    vmargin = check_scalar(vmargin, "vmargin", (float, int), min_val=1)
    hmargin = check_scalar(hmargin, "hmargin", (float, int), min_val=1)

    # apply lower_limit
    if lower_limit is not None:
        adj[abs(adj) < lower_limit] = 0
      
    # analyze network
    G = nx.from_numpy_array(~np.isclose(adj.T, 0), create_using=nx.DiGraph)
    
    dsc_path_length = nx.single_source_shortest_path_length(G, node_index, cutoff=max_dsc)
    del dsc_path_length[node_index]
    
    anc_path_length = nx.single_source_shortest_path_length(
                            G.reverse(), node_index, cutoff=max_anc)
    del anc_path_length[node_index]
    
    # node indices
    dsc_indices = set(dsc_path_length.keys())
    anc_indices = set(anc_path_length.keys())
    isolate_indices = set(nx.isolates(G))
    other_indices = set(np.arange(adj.shape[0])) - dsc_indices - anc_indices \
                                                 - isolate_indices - set([node_index])

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
    graph_attr = {"rankdir":"TB", "splines":"true"}
    graph = graphviz.Digraph("graph", engine="neato", graph_attr=graph_attr)
    
    # colors of clusters
    ds = sorted([d for d in clusters.keys() if d is not None])
    if cmap is None:
        cluster_colors = {d:"black" for d in ds}
    else:
        cmap = plt.get_cmap(cmap)
        cs = np.linspace(0, 1, len(ds) + 1)
        cluster_colors = dict(zip(ds, cmap(cs)))
    cluster_colors[None] = cluster_colors[0]
    
    # distance -> y_position
    pos_y_map = {0:0, None:0}
    
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
    
    node_ids = {node:[] for node in np.arange(adj.shape[0])}
    
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
            nodes = sorted(nodes, key=lambda x: (x in isolate_indices, x in other_indices, x))
        else:
            cls_label = f"ancestor {abs(dist)}" if dist < 0 else f"descendant {abs(dist)}"
            offset_x = 0
            nodes = sorted(nodes)
        pos_y = pos_y_map[dist]
        color = mcolors.to_hex(cluster_colors[dist], keep_alpha=True)
        cls_name = get_cluster_name(dist)

        graph_attr = {"style":"dashed", "label":cls_label, "labeljust":"l", "color":color}
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
                c.node(node_id, node_names[node], pos=f"{pos_x},{pos_y}!", color=color, **kwargs)
                node_ids[node].append((node_id, dist))

            # dummy at the upper left
            x = -cls_margin_l + offset_x
            y = pos_y + cls_margin_t
            c.node(f"{cls_name}-dummy_upper", "", pos=f"{x},{y}!", style="invis",
                                                    fixedsize="true", width="0", height="0")
            
            # dummy at the lower right
            x = (len(nodes) - 1) * hmargin + cls_margin_r + offset_x
            y = pos_y - cls_margin_b
            c.node(f"{cls_name}-dummy_lower", "", pos=f"{x},{y}!", style="invis",
                                                    fixedsize="true", width="0", height="0")

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
    lr = LinearRegression()
    lr.fit(X[:, predictors], X[:, target])
    weight = np.power(np.abs(lr.coef_), gamma)
    reg = LassoLarsIC(criterion="bic")
    reg.fit(X[:, predictors] * weight, X[:, target])
    return reg.coef_ * weight


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
