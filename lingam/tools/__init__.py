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
    "BaseMultipleImputation",
    "BaseMultiGroupCDModel",
]


def bootstrap_with_imputation(
    X,
    n_sampling,
    n_repeats=10,
    imp=None,
    cd_model=None,
    prior_knowledge=None,
    apply_prior_knowledge_softly=False,
    random_state=None
):
    """Discovering causal relations in data with missing values..

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
        The number of bootstraps.
    n_repeats : int, optional (default=10)
        The number of times to complete missing values for each bootstrap sample.
        This value is only used when imp is None.
    imp : object, optional (default=None)
        Instance of a class inheriting from ``BaseMultipleImputation`` class.
        If None, this function uses ``_DefaultMultipleImputation`` to impute datasets.
    cd_model : object, optional (default=None)
        Instance of a class inheriting from ``BaseMultiGroupCDModel`` class.
        If None, this function uses ``MultiGroupDirectLiNGAM`` to estimate the causal order.
    prior_knowledge : array-like, shape (n_features, n_features), optional (default=None)
        Prior knowledge used for the causal discovery, where ``n_features`` is the number of features.
        prior_knowledge is used only if cd_model is None.

        The elements of prior knowledge matrix are defined as follows:

        * ``0`` : :math:`x_i` does not have a directed path to :math:`x_j`
        * ``1`` : :math:`x_i` has a directed path to :math:`x_j`
        * ``-1`` : No prior knowledge is available to know if either of the two cases above (0 or 1) is true.
    apply_prior_knowledge_softly : boolean, optional (default=False)
        If True, apply prior knowledge softly.
        ``apply_prior_knowledge_softly`` is used only if ``cd_model`` is None.
    random_state : int, optional (default=None)
        ``random_state`` is the seed used by the random number generator.

    Returns
    -------
    causal_orders : array-like, shape (n_sampling, n_features)
        The causal order of the fitted model, where
        n_features is the number of features.
    adj_matrices_list : array-like, shape (n_sampling, n_repeats, n_features, n_features)
        The list of adjacency matrices.
    resampled_indices_ : array-like, shape (n_sampling, n_samples)
        The list of original index of resampled samples.
    imputation_results : array-like, shape (n_sampling, n_repeats, n_samples, n_features)
        This array shows the result of the imputation.
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

    if imp is not None and not isinstance(imp, BaseMultipleImputation):
        raise ValueError("imp must be an instance of a subclass of BaseMultipleImputation.")

    n_samples, n_features = X.shape
    if prior_knowledge is not None:
        prior_knowledge = check_array(prior_knowledge)
        if prior_knowledge.shape != (n_features, n_features):
            raise ValueError("The shape of prior_knowledge must be (n_features, n_features).")

    random_state = check_random_state(random_state)

    if imp is None:
        imp = _DefaultMultipleImputation(n_repeats, random_state)

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
        datasets = imp.fit_transform(bootstrap_sample)
        datasets = _check_imputer_outout(datasets, n_samples, n_features)

        n_repeats_impl = len(datasets)

        # run causal discovery assuming a common causal structure
        result = cd_model.fit(datasets)
        causal_order, adjacency_matrices = _check_cd_output(result, n_repeats_impl, n_features)

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


class BaseMultipleImputation(metaclass=ABCMeta):
    """ The abstract class of the causal discovery model for the multigroup data

    Inherit this abstract class and send that instance to ``bootstrap_with_imputation``
    if you need to customize the causal discovery model in ``bootstrap_with_imputation``.
    """

    @abstractmethod
    def fit_transform(self, X):
        """
        This method is called to fit imputed bootstrap sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Target data, where ``n_samples`` is the number of samples
            and ``n_features`` is the number of features.
            ``X`` may contain missing values.

        Returns
        -------
        X_list: list, shape [X, ...]
            The list of imputed X.
        """
        raise NotImplementedError


def _check_imputer_outout(imp_output, n_samples, n_features):
    try:
        imputed_data = check_array(imp_output, allow_nd=True)
    except Exception as e:
        raise ValueError("The return value of imp violates its specification: " + str(e))
    
    if imputed_data.shape[1:] != (n_samples, n_features):
        raise ValueError("The shape of the return value of imp must be (n_repeats, n_samples, n_fatures).")

    imputed_data = list(imputed_data)

    return imputed_data


class BaseMultiGroupCDModel(metaclass=ABCMeta):
    """ The abstract class of the causal discovery model for the multigroup data

    Inherit this abstract class and send that instance to ``bootstrap_with_imputation``
    if you need to customize the causal discovery model in ``bootstrap_with_imputation``.
    """

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
            ``X`` may contain missing values.

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
            ``X_list`` is a list of the imputed bootstrap samples.
            Multiple datasets for training, where ``X`` is an dataset.
            The shape of ''X'' is (n_samples, n_features),
            where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
            Each ``X`` may contain missing values.

        Returns
        -------
        causal_order : array-like
            The estimated causal order.
        adjacenyc_matrices : array-like
            The estimated adjacency matrices.
        """
        raise NotImplementedError


def _check_cd_output(cd_output, n_repeats, n_features):
    if not isinstance(cd_output, tuple) or len(cd_output) != 2:
        raise ValueError("The return value of cd_model.fit() must be a tuple like (causal_order, adjacenecy_matrices).")

    causal_order, adjacency_matrices = cd_output

    try:
        causal_order = check_array(causal_order, ensure_2d=False)
    except Exception as e:
        raise ValueError("causal_order, the output of cd_model, violates its specification: " + str(e))

    if len(causal_order) != n_features:
        raise ValueError("The length of causal_order, the output of cd_model, must be equal to n_features.")

    if not np.array_equal(np.unique(causal_order), np.arange(len(causal_order))):
        raise ValueError("Elements of causal_order, the output of cd_model, must be unique and must be indicates a column number.")

    try:
        adjacency_matrices = check_array(adjacency_matrices, allow_nd=True)
    except Exception as e:
        raise ValueError("adjacency_matrices, the output of cd_model, violates its specification: " + str(e))

    if adjacency_matrices.shape[-2:] != (n_features, n_features):
        raise ValueError("The shape of elements of adjacency_matrices, the output of cd_model, must be (n_features, n_features)")

    return causal_order, adjacency_matrices


class _DefaultMultipleImputation(metaclass=ABCMeta):
    """ The default class for the multiple imputation """

    def __init__(self, n_repeats, random_state):
        self._imp = IterativeImputer(sample_posterior=True, random_state=random_state)
        self._n_repeats = n_repeats

    def fit_transform(self, X):
        X_list = []
        for i in range(self._n_repeats):
            X_ = self._imp.fit_transform(X)
            X_list.append(X_)

        return X_list


class _DefaultMultiGroupCDModel(BaseMultiGroupCDModel):
    """ The default class for the causal discovery on the multigroup data """

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
