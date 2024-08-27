"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""
import numbers
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.utils import check_array, check_scalar, check_random_state
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from ..multi_group_direct_lingam import MultiGroupDirectLiNGAM

__all__ = [
    "bootstrap_with_imputation",
    "BaseMultiGroupCDModel",
]


def bootstrap_with_imputation(
    X,
    n_sampling,
    n_repeats,
    cd_model=None,
    prior_knowledge=None,
    apply_prior_knowledge_softly=False,
    random_state=None
):
    """Discovering causal relations in data which has NaNs.

    `bootstrap_with_imputation` is a function to perform a causal discovery
    on a dataset with missing values. `bootstrap_with_imputation` creates
    `n_sampling` bootstrap samples from the dataset, creates `n_repeats` samples
    for each bootstrap sample, completes the missing values in each sample,
    and runs a causal discovery assuming a common causal structure for
    `n_repeats` samples.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where ``n_samples`` is the number of samples
        and ``n_features`` is the number of features.
    n_sampling : int
        Number of bootstraps.
    n_repeats : int
        Number of times to complete missing values for each bootstrap sample.
    cd_model : object, optional (default=None)
        Instance of a class inheriting from BaseMultiGroupCDModel class.
        If None, use MultiGroupDirectLiNGAM to estimate causal order.
    prior_knowledge : array-like, shape (n_features, n_features), optional (default=None)
        Prior knowledge used for causal discovery, where ``n_features`` is the number of features.
        prior_knowledge is used only if cd_model is None.

        The elements of prior knowledge matrix are defined as follows:

        * ``0`` : :math:`x_i` does not have a directed path to :math:`x_j`
        * ``1`` : :math:`x_i` has a directed path to :math:`x_j`
        * ``-1`` : No prior knowledge is available to know if either of the two cases above (0 or 1) is true.
    apply_prior_knowledge_softly : boolean, optional (default=False)
        If True, apply prior knowledge softly.
        apply_prior_knowledge_softly is used only if cd_model is None.
    random_state : int, optional (default=None)
        ``random_state`` is the seed used by the random number generator.

    Returns
    -------
    causal_orders : array-like, shape (n_sampling, n_features)
        The causal order of fitted model, where
        n_features is the number of features.
    adj_matrices_list : array-like, shape (n_sampling, n_repeats, n_features, n_features)
        The list of adjacency matrices.
    resampled_indices_ : array-like, shape (n_sampling, n_samples)
        The list of original index of resampled samples.
    imputation_results : array-like, shape (n_sampling, n_repeats, n_samples, n_features)
        This array shows a result of the imputation.
        Elements which are not NaN are the imputation values.
    """
    # check args
    X = check_array(X, force_all_finite='allow-nan')

    n_sampling = check_scalar(
        n_sampling,
        "n_sampling",
        (numbers.Integral, np.integer),
        min_val=1
    )

    n_repeats = check_scalar(
        n_repeats,
        "n_repeats",
        (numbers.Integral, np.integer),
        min_val=1
    )

    if cd_model is not None and not isinstance(cd_model, BaseMultiGroupCDModel):
        raise ValueError("cd_model must be an instance of a subclass of BaseMultiGroupCDModel.")

    n_samples, n_features = X.shape
    if prior_knowledge is not None:
        prior_knowledge = check_array(prior_knowledge)
        if prior_knowledge.shape != (n_features, n_features):
            raise ValueError("The shape of prior_knowledge must be (n_features, n_features).")

    random_state = check_random_state(random_state)

    imp = IterativeImputer(sample_posterior=True, random_state=random_state)

    if cd_model is None:
        cd_model = _DefaultMultiGroupCDModel(
            prior_knowledge=prior_knowledge,
            apply_prior_knowledge_softly=apply_prior_knowledge_softly,
            random_state=random_state
        )

    resampled_indices = []
    causal_orders = []
    adj_matrices_list = []
    imputation_results = []

    for i in range(n_sampling):
        # make a bootstrap sample
        resampled_index = random_state.choice(np.arange(X.shape[0]), replace=True, size=len(X))
        bootstrap_sample = X[resampled_index]

        # send bootstrap_sample that has not been imputed yet
        cd_model.before_imputation(bootstrap_sample)

        # make datasets
        datasets = []
        for i in range(n_repeats):
            dataset = imp.fit_transform(bootstrap_sample)
            datasets.append(dataset)

        # run causal discovery assuming a common causal structure
        causal_order, adjacency_matrices = cd_model.fit(datasets)

        # store imputation results
        # hold values only if NaN
        datasets = np.array(datasets)
        imputation_result = np.full(datasets.shape, np.nan)
        pos = np.isnan(bootstrap_sample)
        imputation_result[:, pos] = datasets[:, pos]

        resampled_indices.append(resampled_index)
        causal_orders.append(causal_order)
        adj_matrices_list.append(adjacency_matrices)
        imputation_results.append(imputation_result)

    resampled_indices = np.array(resampled_indices)
    causal_orders = np.array(causal_orders)
    adj_matrices_list = np.array(adj_matrices_list)
    imputation_results = np.array(imputation_results)

    return causal_orders, adj_matrices_list, resampled_indices, imputation_results


class BaseMultiGroupCDModel(metaclass=ABCMeta):

    @abstractmethod
    def before_imputation(self, X):
        """
        This method is called just before the bootstrap sample is imputed.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            ``X`` is a bootstrap sample and has not yet been imputed.
            ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.

        Returns
        -------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, X_list):
        """
        This method is called to fit imputed bootstrap sample.

        Parameters
        ----------
        X_list : list, shape [X, ...]
            ``X_list`` is a list of imputed bootstrap samples.
            Multiple datasets for training, where ``X`` is an dataset.
            The shape of ''X'' is (n_samples, n_features),
            where ``n_samples`` is the number of samples and ``n_features`` is the number of features.

        Returns
        -------
        causal_order : array-like
        adjacenyc_matrices : array-like
        """
        raise NotImplementedError


class _DefaultMultiGroupCDModel(BaseMultiGroupCDModel):

    def __init__(self, prior_knowledge=None, apply_prior_knowledge_softly=False, random_state=None):
        self._model = MultiGroupDirectLiNGAM(
            prior_knowledge=prior_knowledge,
            apply_prior_knowledge_softly=apply_prior_knowledge_softly,
            random_state=random_state
        )

    def before_imputation(self, X):
        pass

    def fit(self, X_list):
        self._model.fit(X_list)
        return self._model.causal_order_, self._model.adjacency_matrices_
