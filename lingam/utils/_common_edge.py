import numpy as np
import graphviz
import networkx as nx
from matplotlib.colors import is_color_like
from sklearn.utils import check_array, check_scalar


def get_common_edge_probabilities(bootstrap_results, mode='across'):
    """
    Calculate the probability of common edges across multiple bootstrap results.

    This function analyzes adjacency matrices from a list of bootstrap results and computes
    the probability of common edges (non-zero and not NaN) and hidden common cause edges (NaN)
    either per sample or across all samples depending on the selected mode.

    Parameters
    ----------
    bootstrap_results : list
        A list of objects, each containing an attribute `adjacency_matrices_`, which is a list
        or array of adjacency matrices (2D numpy arrays) from bootstrap iterations.

    mode : str, optional
        The mode of calculation. Must be either:

        * 'per': Computes common edge probabilities by extracting common edges at each bootstrap iteration.
        * 'across': Computes common edge probabilities across all samples within each bootstrap result.

    Returns
    -------
    common_edge_probabilities : np.ndarray
        A stacked array of shape (2, n, n), where:

        * common_edge_probabilities[0]: The probabilities of common edges.
        * common_edge_probabilities[1]: The nan probabilities of hidden common cause.
    """
    # check parameters
    if not isinstance(bootstrap_results, list):
        raise ValueError("bootstrap_results must be a list.")

    for bootstrap_result in bootstrap_results:
        if not hasattr(bootstrap_result, "adjacency_matrices_"):
            raise AttributeError("The bootstrap_result must implement a callable 'adjacency_matrices_'.")

    if mode == 'across':
        common_edge_prob_list = []
        hidden_common_edge_prob_list = []

        for bootstrap_result in bootstrap_results:
            matrices = np.array(bootstrap_result.adjacency_matrices_)

            # common edge: non-zero and not NaN
            valid_mask = ~np.isnan(matrices)
            nonzero_mask = (matrices != 0) & valid_mask
            common_edge_prob = nonzero_mask.sum(axis=0) / len(matrices)
            common_edge_prob_list.append(common_edge_prob)

            # hidden common cause edges: NaN
            unobserved_prob = np.isnan(matrices).sum(axis=0) / len(matrices)
            hidden_common_edge_prob_list.append(unobserved_prob)

        # Combine probabilities
        common_edge_prob = np.prod(common_edge_prob_list, axis=0)
        hidden_common_edge_prob = np.prod(hidden_common_edge_prob_list, axis=0)

        # Stack into a single array of shape (2, n, n)
        common_edge_probabilities = np.stack((common_edge_prob, hidden_common_edge_prob), axis=0)

    elif mode == 'per':
        n_samples = len(bootstrap_results[0].adjacency_matrices_)
        n_vars = bootstrap_results[0].adjacency_matrices_[0].shape[0]
        common_edge_counts = np.zeros((n_vars, n_vars))
        hidden_common_edge_counts = np.zeros((n_vars, n_vars))

        # Extract common edges from each bootstrap iteration
        for i in range(n_samples):
            nonzero_masks = []
            nan_masks = []
            for bootstrap_result in bootstrap_results:
                adj_matrix = bootstrap_result.adjacency_matrices_[i]

                # common edge: non-zero and not NaN
                valid_mask = ~np.isnan(adj_matrix)
                nonzero_mask = (adj_matrix != 0) & valid_mask
                nonzero_masks.append(nonzero_mask)

                # hidden common cause edges: NaN
                nan_mask = np.isnan(adj_matrix)
                nan_masks.append(nan_mask)

            common_nonzero_mask = np.logical_and.reduce(nonzero_masks)
            common_nan_mask = np.logical_and.reduce(nan_masks)
            common_edge_counts += common_nonzero_mask.astype(int)
            hidden_common_edge_counts += common_nan_mask.astype(int)

        # Combine probabilities
        common_edge_prob = common_edge_counts / n_samples
        hidden_common_edge_prob = hidden_common_edge_counts / n_samples

        # Stack into a single array of shape (2, n, n)
        common_edge_probabilities = np.stack((common_edge_prob, hidden_common_edge_prob), axis=0)

    else:
        raise ValueError("Invalid mode. Choose 'per' or 'across'.")

    return common_edge_probabilities


def print_common_edge_directions(common_edge_probabilities, confidence_level=0.01, labels=None):
    """
    Print statistically significant causal directions based on common edge probabilities.

    Parameters
    ----------
    common_edge_probabilities : ndarray
        A 2D NumPy array where each element [i, j] represents the probability
        that variable j causes variable i. The matrix is expected to be square
        and contain non-negative values.

    confidence_level : float, optional
        A threshold between 0 and 1 (inclusive) used to determine whether a causal
        direction is considered statistically significant. Only edges with probabilities
        strictly greater than this value will be displayed.

    labels : list of str, optional
        A list of variable names corresponding to the indices in the matrix.
        If provided, these labels will be used in the output instead of generic
        variable names (e.g., x0, x1, ...).
    """
    # check parameters
    if not 0.0 <= confidence_level < 1.0:
        raise ValueError("confidence_level must be a value between 0 and 1.")

    if not isinstance(common_edge_probabilities, np.ndarray):
        raise TypeError("common_edge_probabilities must be a NumPy ndarray.")

    if common_edge_probabilities.ndim != 3:
        raise ValueError("common_edge_probabilities must be a 3-dimensional array.")

    if common_edge_probabilities.shape[0] != 2:
        raise ValueError("The first dimension of common_edge_probabilities must be 2 (presence and unobserved probabilities).")

    n_rows, n_cols = common_edge_probabilities.shape[1], common_edge_probabilities.shape[2]

    # Probability matrix of shared edges
    joint_presence_prob = common_edge_probabilities[0]
    directions = []
    for from_ in range(n_cols):
        for to in range(n_rows):
            prob = joint_presence_prob[to, from_]
            if prob > confidence_level:
                directions.append((from_, to, prob, "<---"))

    # Probability matrix of NaNs due to shared unobserved common causes
    joint_unobserved_prob = common_edge_probabilities[1]
    bidirectional = set()
    for from_ in range(n_cols):
        for to in range(n_rows):
            if (to, from_) in bidirectional:
                continue
            prob = joint_unobserved_prob[to, from_]
            if prob > confidence_level:
                # In VAR-LiNGAM, due to mismatched matrix dimensions
                if from_ < n_rows:
                    reverse_prob = joint_unobserved_prob[from_, to]
                    if prob == reverse_prob:
                        bidirectional.add((to, from_))
                        bidirectional.add((from_, to))
                directions.append((from_, to, prob, "<-->"))

    directions.sort(key=lambda x: -x[2])

    for (from_, to, prob, edge_d) in directions:
        if labels:
            print(f"{labels[to]} {edge_d} {labels[from_]} ({100*prob:.1f}%)")
        else:
            print(f"x{to} {edge_d} x{from_} ({100*prob:.1f}%)")


def make_dot_for_nan_probability_matrix(
    adjacency_matrix,
    labels=None,
    lower_limit=0.01,
    path=None,
    path_color=None,
    ignore_shape=False,
):
    """Directed graph source code in the DOT language with specified nan probability matrix of hidden common cause.

    Parameters
    ----------
    adjacency_matrix : array-like with shape (n_features, n_features)
        Nan probability matrix of hidden common cause to make graph, where ``n_features`` is the number of features.
    labels : array-like, optional (default=None)
        Label to use for graph features.
    lower_limit : float, optional (default=0.01)
        Threshold for drawing direction.
        If float, then directions with absolute values of coefficients less than ``lower_limit`` are excluded.
    path : tuple, optional (default=None)
        Path to highlight. Tuple of start index and end index.
    path_color : string, optional (default=None)
        Colors to highlight a path.
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

    idx = np.abs(B) > lower_limit

    nx_graph = None
    if path is not None:
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
    # This matrix is a probability matrix of nan, and the diagonal components contain the same value
    dirs = np.where(np.tril(idx))
    for to, from_, coef in zip(dirs[0], dirs[1], B[dirs]):
        kwargs = {}
        if path is not None and (from_, to) in path_edges:
            kwargs["penwidth"] = "2"
            kwargs["fontcolor"] = path_color
            kwargs["color"] = path_color
        if from_ != to:
            if idx[to, from_] and B[from_, to] == B[to, from_]:
                d.edge(names[from_], names[to], dir="both", label=f"{coef:.2f}", **kwargs)

    return d
