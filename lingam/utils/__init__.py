"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/site/sshimizu06/lingam
"""
import graphviz
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LassoLarsIC, LinearRegression
from sklearn.utils import check_array

import igraph as ig
from scipy.special import expit as sigmoid
import random


__all__ = [
    "print_causal_directions",
    "print_dagc",
    "make_prior_knowledge",
    "remove_effect",
    "make_dot",
    "predict_adaptive_lasso",
    "get_sink_variables",
    "get_exo_variables",
    "find_all_paths",
    "simulate_dag",
    "simulate_parameter",
    "simulate_linear_sem",
    "count_accuracy",
    "set_random_seed",
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
            Data for the  specified variable.
        """
        if sem_type == 'gauss':
            z = np.random.normal(scale=noise_scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=noise_scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=noise_scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        elif sem_type == 'subGaussian':
            z = np.random.normal(scale=noise_scale, size=n)
            q = 0.5 + 0.3 * np.random.rand(1)  # sub-Gaussian
            z = np.sign(z) * pow(np.abs(z), q)
            z = z - np.mean(z)
            z = z / np.std(z)
            x = X @ w + z
        elif sem_type == 'supGaussian':
            z = np.random.normal(scale=noise_scale, size=n)
            q = 1.2 + 0.8 * np.random.rand(1)  # super-Gaussian
            z = np.sign(z) * pow(np.abs(z), q)
            z = z - np.mean(z)
            z = z / np.std(z)
            x = X @ w + z
        elif sem_type == 'nonGaussian':
            z = np.random.normal(scale=noise_scale, size=n)
            qq = -1
            if qq == 1:
                q = 0.5 + 0.3 * np.random.rand(1)  # sub-Gaussian
            else:
                q = 1.2 + 0.8 * np.random.rand(1)  # super-Gaussian
            z = np.sign(z) * pow(np.abs(z), q)
            z = z - np.mean(z)
            z = z / np.std(z)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(0, 1, n)
            z = z - np.mean(z)
            z = z / np.std(z)
            x = X @ w + z
        elif sem_type == 'gamma':
            z = np.random.gamma(2, 2, n)
            z = z - np.mean(z)
            z = z / np.std(z)
            x = X @ w + z
        elif sem_type == 'laplace':
            z = np.random.laplace(0, scale=noise_scale, size=n)
            x = X @ w + z
        else:
            raise ValueError('unknown sem type')
        return x

    n_features = adjacency_matrix.shape[0]
    if np.isinf(n_samples):
        if sem_type == 'gauss':
            # make 1/n_features X'X = true cov
            X = np.sqrt(n_features) * noise_scale * np.linalg.pinv(np.eye(n_features) - adjacency_matrix)
            return X
        else:
            raise ValueError('population risk not available')
    X = np.zeros([n_samples, n_features])

    G = ig.Graph.Weighted_Adjacency(adjacency_matrix.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == n_features

    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], adjacency_matrix[parents, j])
    return X


def count_accuracy(W_true, W, W_und=None):
    """Compute FDR, TPR, and FPR for B, or optionally for CPDAG = B + B_und.

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
    B_true = (W_true != 0)
    B = (W != 0)
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

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=n_features, m=n_edges)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=n_features, m=int(round(n_edges / n_features)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * n_features)
        G = ig.Graph.Random_Bipartite(top, n_features - top, m=n_edges, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
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
            sign = "" if "sign" not in dag else "(b>0)" if dag["sign"][j] > 0 else "(b<0)"
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

    d = graphviz.Digraph(engine="dot")

    # nodes
    names = labels if labels else [f"x{i}" for i in range(len(B))]
    for name in names:
        d.node(name)

    # edges
    idx = np.abs(B) > lower_limit
    dirs = np.where(idx)
    for to, from_, coef in zip(dirs[0], dirs[1], B[idx]):
        d.edge(names[from_], names[to], label=f"{coef:.2f}")

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
