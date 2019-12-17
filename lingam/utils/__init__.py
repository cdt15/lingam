"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/site/sshimizu06/lingam
"""
import numpy as np
from sklearn import linear_model
from sklearn.utils import check_array

__all__ = ['print_causal_directions', 'print_dagc',
           'make_prior_knowledge', 'remove_effect']


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
