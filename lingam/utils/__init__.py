"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/site/sshimizu06/lingam
"""
import numpy as np
from sklearn import linear_model
from sklearn.utils import check_array
import graphviz

__all__ = ['print_causal_directions', 'print_dagc',
           'make_prior_knowledge', 'remove_effect', 'make_dot']


def print_causal_directions(cdc, n_sampling, labels=None):
    """ Print causal directions of bootstrap result to stdout.

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
    for i, (fr, to, co) in enumerate(zip(cdc['from'], cdc['to'], cdc['count'])):
        sign = '' if 'sign' not in cdc else '(b>0)' if cdc['sign'][i] > 0 else '(b<0)'
        if labels:
            print(
                f'{labels[to]} <--- {labels[fr]} {sign} ({100*co/n_sampling:.1f}%)')
        else:
            print(f'x{to} <--- x{fr} {sign} ({100*co/n_sampling:.1f}%)')


def print_dagc(dagc, n_sampling, labels=None):
    """ Print DAGs of bootstrap result to stdout.

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
    for i, (dag, co) in enumerate(zip(dagc['dag'], dagc['count'])):
        print(f'DAG[{i}]: {100*co/n_sampling:.1f}%')
        for j, (fr, to) in enumerate(zip(dag['from'], dag['to'])):
            sign = '' if 'sign' not in dag else '(b>0)' if dag['sign'][j] > 0 else '(b<0)'
            if labels:
                print('\t' + f'{labels[to]} <--- {labels[fr]} {sign}')
            else:
                print('\t' + f'x{to} <--- x{fr} {sign}')


def make_prior_knowledge(n_variables, exogenous_variables=None, sink_variables=None, paths=None, no_paths=None):
    """ Make matrix of prior knowledge.

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
            prior_knowledge[path[0], path[1]] = 0
    if sink_variables:
        for var in sink_variables:
            prior_knowledge[:, var] = 0
    if exogenous_variables:
        for var in exogenous_variables:
            prior_knowledge[var, :] = 0
    np.fill_diagonal(prior_knowledge, 0)
    return prior_knowledge


def remove_effect(X, remove_features):
    """ Create a dataset that removes the effects of features by linear regression.

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


def make_dot(adjacency_matrix, labels=None, lower_limit=0.01,
             prediction_feature_indices=None, prediction_target_label='Y(pred)',
             prediction_line_color='red',
             prediction_coefs=None, prediction_feature_importance=None, ignore_shape=False):
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
    ignore_shape : boolean
        Ignore checking the shape of adjaceny_matrix or not.

    Returns
    -------
    graph : graphviz.Digraph
        Directed graph source code in the DOT language.
    """
    # Check parameters
    B = check_array(adjacency_matrix)
    if not ignore_shape and B.shape[0] != B.shape[1]:
        raise ValueError("'adjacency_matrix' is not square matrix.")
    if labels is not None:
        if B.shape[1] != len(labels):
            raise ValueError(
                "Length of 'labels' does not match length of 'adjacency_matrix'")
    if prediction_feature_indices is not None:
        if prediction_coefs is not None and (len(prediction_feature_indices) != len(prediction_coefs)):
            raise ValueError(
                "Length of 'prediction_coefs' does not match length of 'prediction_feature_indices'")
        if prediction_feature_importance is not None and (len(prediction_feature_indices) != len(prediction_feature_importance)):
            raise ValueError(
                "Length of 'prediction_feature_importance' does not match length of 'prediction_feature_indices'")

    d = graphviz.Digraph(engine='dot')

    # nodes
    names = labels if labels else [f'x{i}' for i in range(len(B))]
    for name in names:
        d.node(name)

    # edges
    idx = np.abs(B) > lower_limit
    dirs = np.where(idx)
    for to, from_, coef in zip(dirs[0], dirs[1], B[idx]):
        d.edge(names[from_], names[to], label=f'{coef:.2f}')

    # integrate of prediction model
    if prediction_feature_indices is not None:
        d.node(prediction_target_label,
               color=prediction_line_color,
               fontcolor=prediction_line_color)

        if prediction_coefs is not None:
            for from_, coef in zip(prediction_feature_indices, prediction_coefs):
                if np.abs(coef) > lower_limit:
                    d.edge(names[from_],
                           prediction_target_label,
                           label=f'{coef:.2f}',
                           color=prediction_line_color,
                           fontcolor=prediction_line_color,
                           style='dashed')

        elif prediction_feature_importance is not None:
            for from_, imp in zip(prediction_feature_indices, prediction_feature_importance):
                d.edge(names[from_],
                       prediction_target_label,
                       label=f'({imp})',
                       color=prediction_line_color,
                       fontcolor=prediction_line_color,
                       style='dashed')

        else:
            for from_ in prediction_feature_indices:
                d.edge(names[from_],
                       prediction_target_label,
                       color=prediction_line_color,
                       style='dashed')

    return d
