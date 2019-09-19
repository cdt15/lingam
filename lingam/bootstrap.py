"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/site/sshimizu06/lingam
"""

import numbers
import numpy as np
from sklearn.utils import check_array, resample


class BootstrapMixin():
    """Mixin class for all LiNGAM algorithms that implement the method of bootstrapping."""

    def bootstrap(self, X, n_sampling):
        """Evaluate the statistical reliability of DAG based on the bootstrapping.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.
        n_sampling : int
            Number of bootstrapping samples.

        Returns
        -------
        result : BootstrapResult
            Returns the result of bootstrapping.
        """
        # Check parameters
        X = check_array(X)

        if isinstance(n_sampling, (numbers.Integral, np.integer)):
            if not 0 < n_sampling:
                raise ValueError(
                    'n_sampling must be an integer greater than 0.')
        else:
            raise ValueError('n_sampling must be an integer greater than 0.')

        # Bootstrapping
        adjacency_matrices = []
        for _ in range(n_sampling):
            model = self.fit(resample(X, random_state=self._random_state))
            adjacency_matrices.append(model.adjacency_matrix_)
        return BootstrapResult(adjacency_matrices)


class BootstrapResult(object):
    """The result of bootstrapping."""

    def __init__(self, adjacency_matrices):
        """Construct a BootstrapResult.

        Parameters
        ----------
        adjacency_matrices : array-like, shape (n_sampling)
            The adjacency matrix list by bootstrapping.
        """
        self._adjacency_matrices = adjacency_matrices

    @property
    def adjacency_matrices_(self):
        """The adjacency matrix list by bootstrapping.

        Returns
        -------
        adjacency_matrices_ : array-like, shape (n_sampling)
            The adjacency matrix list, where ``n_sampling`` is 
            the number of bootstrap sampling.
        """
        return self._adjacency_matrices

    def get_causal_direction_counts(self, n_directions=None, min_causal_effect=None):
        """Get causal direction count as a result of bootstrapping.

        Parameters
        ----------
        n_directions : int, optional (default=None)
            If int, then The top ``n_directions`` items are included in the result
        min_causal_effect : float, optional (default=None)
            Threshold for detecting causal direction. 
            If float, then causal directions with absolute values of causal effects less than ``min_causal_effect`` are excluded.

        Returns
        -------
        causal_direction_counts : dict
            List of causal directions sorted by count in descending order. 
            The dictionary has the following format:: 

            {'from': [n_directions], 'to': [n_directions], 'count': [n_directions]}

            where ``n_directions`` is the number of causal directions.
        """
        # Check parameters
        if isinstance(n_directions, (numbers.Integral, np.integer)):
            if not 0 < n_directions:
                raise ValueError(
                    'n_directions must be an integer greater than 0')
        elif n_directions is None:
            pass
        else:
            raise ValueError('n_directions must be an integer greater than 0')

        if min_causal_effect is None:
            min_causal_effect = 0.0
        else:
            if not 0.0 < min_causal_effect:
                raise ValueError('min_causal_effect must be an value greater than 0.')

        # Count causal directions
        directions = [
            np.array(np.where(np.abs(am) > min_causal_effect)).T for am in self._adjacency_matrices]
        directions = np.concatenate(directions)

        if len(directions) == 0:
            return {'from': [], 'to': [], 'count': []}

        directions, counts = np.unique(directions, axis=0, return_counts=True)
        sort_order = np.argsort(-counts)
        sort_order = sort_order[:n_directions] if n_directions is not None else sort_order
        counts = counts[sort_order]
        directions = directions[sort_order]

        return {
            'from': directions[:, 1].tolist(),
            'to': directions[:, 0].tolist(),
            'count': counts.tolist()
        }

    def get_directed_acyclic_graph_counts(self, n_dags=None, min_causal_effect=None):
        """Get DAGs count as a result of bootstrapping.

        Parameters
        ----------
        n_dags : int, optional (default=None)
            If int, then The top ``n_dags`` items are included in the result
        min_causal_effect : float, optional (default=None)
            Threshold for detecting causal direction. 
            If float, then causal directions with absolute values of causal effects less than ``min_causal_effect`` are excluded.

        Returns
        -------
        directed_acyclic_graph_counts : dict
            List of directed acyclic graphs graphs sorted by count in descending order. 
            The dictionary has the following format:: 

            {'dag': [n_dags], 'count': [n_dags]}.

            where ``n_dags`` is the number of directed acyclic graphs.
        """
        # Check parameters
        if isinstance(n_dags, (numbers.Integral, np.integer)):
            if not 0 < n_dags:
                raise ValueError('n_dags must be an integer greater than 0')
        elif n_dags is None:
            pass
        else:
            raise ValueError('n_dags must be an integer greater than 0')

        if min_causal_effect is None:
            min_causal_effect = 0.0
        else:
            if not 0.0 < min_causal_effect:
                raise ValueError('min_causal_effect must be an value greater than 0.')

        # Count directed acyclic graphs
        dags = [np.abs(am) > min_causal_effect for am in self._adjacency_matrices]
        dags, counts = np.unique(dags, axis=0, return_counts=True)
        sort_order = np.argsort(-counts)
        sort_order = sort_order[:n_dags] if n_dags is not None else sort_order
        counts = counts[sort_order]
        dags = dags[sort_order]

        dags = [{
            'from': np.where(dag)[1].tolist(),
            'to': np.where(dag)[0].tolist()} for dag in dags]

        return {
            'dag': dags,
            'count': counts.tolist()
        }
