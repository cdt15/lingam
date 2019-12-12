"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/site/sshimizu06/lingam
"""
import numbers
import numpy as np
from sklearn.utils import check_array, resample

from .direct_lingam import DirectLiNGAM
from .bootstrap import BootstrapResult


class MultiGroupDirectLiNGAM(DirectLiNGAM):
    """Implementation of DirectLiNGAM Algorithm with multiple groups [1]_

    References
    ----------
    .. [1] S. Shimizu. Joint estimation of linear non-Gaussian acyclic models. Neurocomputing, 81: 104-107, 2012. 
    """

    def __init__(self, random_state=None, prior_knowledge=None):
        """Construct a model.

        Parameters
        ----------
        random_state : int, optional (default=None)
            ``random_state`` is the seed used by the random number generator.
        prior_knowledge : array-like, shape (n_features, n_features), optional (default=None)
            Prior knowledge used for causal discovery, where ``n_features`` is the number of features.
        """
        super().__init__(random_state)
        self._prior_knowledge = prior_knowledge

    def fit(self, X_list):
        """Fit the model to multiple datasets.

        Parameters
        ----------
        X_list : list, shape [X, ...]
            Multiple datasets for training, where ``X`` is an dataset.
            The shape of ''X'' is (n_samples, n_features), 
            where ``n_samples`` is the number of samples and ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Check parameters
        if not isinstance(X_list, list):
            raise ValueError('X_list must be a list.')

        if len(X_list) < 2:
            raise ValueError('X_list must be a list containing at least two items')

        n_features = check_array(X_list[0]).shape[1]
        X_list_ = []
        for X in X_list:
            X_ = check_array(X)
            if X_.shape[1] != n_features:
                raise ValueError('X_list must be a list with the same number of features')
            X_list_.append(X_)
        X_list = np.array(X_list_)

        if self._prior_knowledge is not None:
            self._Aknw = check_array(self._prior_knowledge)
            self._Aknw = np.where(self._Aknw < 0, np.nan, self._Aknw)
            if (n_features, n_features) != self._Aknw.shape:
                raise ValueError('The shape of prior knowledge must be (n_features, n_features)')
        else:
            self._Aknw = None

        # Causal discovery
        U = np.arange(n_features)
        K = []
        X_list_ = [np.copy(X) for X in X_list]
        for _ in range(n_features):
            m = self._search_causal_order(X_list_, U)
            for i in U:
                if i != m:
                    for d in range(len(X_list_)):
                        X_list_[d][:, i] = self._residual(X_list_[d][:, i], X_list_[d][:, m])
            K.append(m)
            U = U[U != m]

        self._causal_order = K

        self._adjacency_matrices = []
        for X in X_list:
            self._estimate_adjacency_matrix(X)
            self._adjacency_matrices.append(self._adjacency_matrix)
        return self

    def bootstrap(self, X_list, n_sampling):
        """Evaluate the statistical reliability of DAG based on the bootstrapping.

        Parameters
        ----------
        X_list : array-like, shape (X, ...)
            Multiple datasets for training, where ``X`` is an dataset.
            The shape of ''X'' is (n_samples, n_features), 
            where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
        n_sampling : int
            Number of bootstrapping samples.

        Returns
        -------
        results : array-like, shape (BootstrapResult, ...)
            Returns the results of bootstrapping for multiple datasets.
        """
        if len(X_list) < 2:
            raise ValueError('X_list must be a list containing at least two items')

        n_features = check_array(X_list[0]).shape[1]
        X_list_ = []
        for X in X_list:
            X_ = check_array(X)
            if X_.shape[1] != n_features:
                raise ValueError('X_list must be a list with the same number of features')
            X_list_.append(X_)
        X_list = np.array(X_list_)

        if isinstance(n_sampling, (numbers.Integral, np.integer)):
            if not 0 < n_sampling:
                raise ValueError('n_sampling must be an integer greater than 0.')
        else:
            raise ValueError('n_sampling must be an integer greater than 0.')

        # Bootstrapping
        adjacency_matrices_list = [[] for _ in range(X_list.shape[0])]
        for _ in range(n_sampling):
            resampled_X_list = [resample(X) for X in X_list]
            model = self.fit(resampled_X_list)
            for i, am in enumerate(model.adjacency_matrices_):
                adjacency_matrices_list[i].append(am)

        result_list = []
        for adjacency_matrices in adjacency_matrices_list:
            result_list.append(BootstrapResult(adjacency_matrices))

        return result_list

    def _search_causal_order(self, X_list, U):
        """Search the causal ordering."""
        Uc, Vj = self._search_candidate(U)
        if len(Uc) == 1:
            return Uc[0]

        total_size = 0
        for X in X_list:
            total_size += len(X)

        MG_list = []
        for i in Uc:
            MG = 0
            for X in X_list:
                M = 0
                for j in U:
                    if i != j:
                        xi_std = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
                        xj_std = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])
                        ri_j = xi_std if i in Vj and j in Uc else self._residual(xi_std, xj_std)
                        rj_i = xj_std if j in Vj and i in Uc else self._residual(xj_std, xi_std)
                        M += np.min([0, self._diff_mutual_info(xi_std, xj_std, ri_j, rj_i)])**2
                MG += M * (len(X) / total_size)
            MG_list.append(-1.0 * MG)
        return Uc[np.argmax(MG_list)]

    @property
    def adjacency_matrices_(self):
        """Estimated adjacency matrices.

        Returns
        -------
        adjacency_matrices_ : array-like, shape (B, ...)
            The list of adjacency matrix B for multiple datasets.
            The shape of B is (n_features, n_features), where 
            n_features is the number of features.
        """
        return self._adjacency_matrices
