from .direct_lingam import DirectLiNGAM
from .ica_lingam import ICALiNGAM
from .bottom_up_parce_lingam import BottomUpParceLiNGAM
from .var_lingam import VARLiNGAM

from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from scipy.special import expit

from sklearn.utils import check_array, check_random_state
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import clone, is_regressor, is_classifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection._search import BaseSearchCV


class CausalBasedSimulator:
    """
    Causal based simulator.

    Attributes
    ----------
    train_result_ : dict
        Information about trained models.
    """

    def train(self, X, causal_graph, cd_algo_name=None, models=None, is_discrete=None):
        """
        Estimate functional relations between variables and variable
        distributions based on the training data ``X`` and the causal graph.
        The functional relations represents by linear regression model
        if the object variable is numeric, and represents by logistic
        regression model if the object variable is discrete by default.
        ``train_result_`` will be exposed after executing train().

        Parameters
        ----------
        X : array-like
            Training data.

        causal_graph : array-like
            Causal graph.

        cd_algo_name : str
            Algorithm type.

        models : dict, default=None
            Dictionary about the model of a variable in X. Models are cloned internaly
            and are trained to infer functioal relations. Given instances of
            the model are cloned to estimate the functional relation between
            variables.

        Returns
        -------
        self : Object
        """

        if cd_algo_name is None:
            cd_algo_name = "DirectLiNGAM"
        if not isinstance(cd_algo_name, str):
            raise TypeError("cd_algo_name must be str or None.")

        impl_constructor = self._dispatch_impl(cd_algo_name)
        impl = impl_constructor(X, causal_graph, is_discrete=is_discrete)

        train_models = self._check_models(models, impl.endog_names_, impl.discrete_endog_names_)

        train_result = self._train(train_models, causal_graph, impl)

        self._impl = impl
        self._train_result = train_result
        return self

    def run(
        self,
        changing_exog=None,
        changing_models=None,
        shuffle_residual=False,
        random_state=None,
    ):
        """
        Generate simulated data using trained models and the given
        causal graph with given exogenous data and models.
        Specifying environmental changes to ``changing_exog`` or
        specifiyig changes in fucitonal relation to ``change_models``
        effects simulated data.
        Residuals to simulate variables are shuffled using radom_state
        if ``shuffle_residual`` is True.

        Parameters
        ----------
        changing_exog : dict, default=None
            Dictioary about exogeous variables which keys are variable
            names and values are data of variables. That variable name
            should be a one of column names of X and the length should
            be same as X.

        changing_model : dict, default=None
            Dictionary about the changing models which elements are dictionary.
            The key is the name of the target variable and that value is
            a dictionary for that model. This dictionary contains three keys,
            parant_names, coef and model. parent_names is the mandatory key and
            whose value is the list of parent names. coef and model are selective.
            coef is the list of the coefficients of the parent variables, which
            must be same lenght as parent_names. The value of model must be
            a trained machine laerning instance.

        shuffle_residual : bool, default=True
            If True, residuals are shuffled.

        random_state : int, default=None
            If shuffle_residual is True, random_state is used as seed.

        Returns
        -------
        simulated_data : pandas.DataFrame
            simulated data.
        """

        if self._train_result is None:
            raise RuntimeError("run() must be executed after train() is executed")

        random_state = check_random_state(random_state)

        # check inputs
        changing_exog = self._check_changing_exog(
            changing_exog,
            self._impl.exog_length_,
            self._impl.endog_names_,
            self._impl.discrete_endog_names_
        )

        changing_models = self._check_changing_models(
            changing_models,
            self._impl.endog_names_,
            self._impl.discrete_endog_names_
        )

        simulated_data = self._simulate(
            changing_exog,
            changing_models,
            self._impl,
            self._train_result,
            shuffle_residual,
            random_state,
        )

        return simulated_data

    @property
    def train_result_(self):
        return self._train_result

    def _check_model_instance(self, model, var_name, discrete_endog_names):
        if var_name not in discrete_endog_names:
            check_model_type = is_regressor
        else:
            check_model_type = is_classifier

        if isinstance(model, Pipeline):
            if not check_model_type(model.steps[-1][-1]):
                mes = "The last step in Pipeline should be an "
                mes += "instance of a regression/classification model."
                raise RuntimeError(mes)
        elif isinstance(model, BaseSearchCV):
            if not check_model_type(model.get_params()["estimator"]):
                mes = "The type of the estimator shall be an "
                mes += "instance of a regression/classification model."
                raise RuntimeError(mes)
        else:
            if not check_model_type(model):
                mes = "The type of the estimator shall be an "
                mes += "instance of a regression/classification model."
                raise RuntimeError(mes)

        if check_model_type == is_classifier:
            try:
                func = getattr(model, "predict_proba")
                if not callable(func):
                    raise Exception
            except Exception:
                mes = "Classification models shall have "
                mes += "predict_proba()."
                raise RuntimeError(mes)

    def _check_models(self, models, endog_names, discrete_endog_names):
        if models is None:
            return {}

        if not isinstance(models, dict):
            raise RuntimeError("models must be a dictionary.")

        for var_name, model in models.items():
            if var_name not in endog_names:
                raise RuntimeError(f"Unknown variable name ({var_name})")

            self._check_model_instance(model, var_name, discrete_endog_names)

        return models

    def _check_changing_exog(self, changing_exog, n_samples, endog_names, discrete_endog_names):
        if changing_exog is None:
            return {}

        if not isinstance(changing_exog, dict):
            raise RuntimeError("changing_exog must be a dictionary.")

        changing_exog_ = {}
        for var_name, values in changing_exog.items():
            if var_name not in endog_names:
                raise RuntimeError(f"Unknown key in changing_exog. ({var_name})")

            if var_name in discrete_endog_names:
                raise RuntimeError(
                    f"Discrete variables shall not be specified. ({var_name})"
                )

            s = check_array(values, ensure_2d=False, dtype=None).ravel()
            if s.shape[0] != n_samples:
                raise RuntimeError(f"Wrong length. ({s.shape[0]} != {n_samples})")

            changing_exog_[var_name] = values

        return changing_exog_

    def _check_changing_models(self, changing_models, endog_names, discrete_endog_names):
        if changing_models is None:
            return {}

        if not isinstance(changing_models, dict):
            raise RuntimeError("changing_models shall be list.")

        changing_models_ = {}
        for target_name, model_info in changing_models.items():
            if not isinstance(model_info, dict):
                raise RuntimeError("changing_models shall be list of dictionaries.")

            # check target_name
            if not isinstance(target_name, str):
                raise TypeError("Key of changing_models must be str.")
            if target_name not in endog_names:
                raise RuntimeError(f"Unknown name. ({target_name})")

            # parent_names key
            if "parent_names" not in model_info.keys():
                raise KeyError("model_info must have parent_names key.")

            parent_names = model_info["parent_names"]
            if not isinstance(parent_names, list):
                raise KeyError("parent_names must be list.")

            for parent_name in model_info["parent_names"]:
                if parent_name not in endog_names:
                    raise RuntimeError(f"Unknown name. ({parent_name})")

            if len(parent_names) == 0:
                changing_models_[target_name] = {"parent_names": []}
                continue

            # coef and model key
            if "coef" not in model_info.keys() and "model" not in model_info.keys():
                raise KeyError("Elements of changing_models must have coef or model key when parent_names is set.")

            # coef key
            if "coef" in model_info.keys():
                coef = check_array(model_info["coef"], ensure_2d=False, ensure_min_samples=0)
                if len(coef) != len(parent_names):
                    raise ValueError("len(coef) != len(parent_names)")

                if target_name not in discrete_endog_names:
                    model = _LinearRegression(coef)
                else:
                    model = _LogisticRegression(coef)

                changing_models_[target_name] = {"parent_names": parent_names, "model": model}
                continue

            # model key
            if "model" not in model_info.keys() or model_info["model"] is None:
                raise KeyError("model must be set when coef isn't set.")

            model = model_info["model"]
            self._check_model_instance(model, target_name, discrete_endog_names)

            changing_models_[target_name] = {"parent_names": parent_names, "model": model}

        return changing_models_

    def _dispatch_impl(self, cd_algo_name):
        if cd_algo_name is None:
            return CBSILiNGAM
        elif cd_algo_name == DirectLiNGAM.__name__:
            return CBSILiNGAM
        elif cd_algo_name == ICALiNGAM.__name__:
            return CBSILiNGAM
        elif cd_algo_name == BottomUpParceLiNGAM.__name__:
            return CBSIUnobsCommonCauseLiNGAM
        elif cd_algo_name == VARLiNGAM.__name__:
            return CBSITimeSeriesLiNGAM
        else:
            raise ValueError("Unknown cd_algo_name")

    def _train(self, models, causal_graph, impl):
        train_result = {}

        for target_name in impl.endog_names_:
            y = impl.get_data(target_name)
            y = y.ravel()

            parent_names = impl.get_parent_names(target_name)
            if len(parent_names) == 0:
                train_result[target_name] = {
                    "model": None,
                    "parent_names": [],
                    "predicted": None,
                    "residual": y,
                }
                continue

            X = impl.get_data(parent_names)

            is_classifier = target_name in impl.discrete_endog_names_

            # select a model to train
            if target_name in models.keys():
                model = clone(models[target_name])
            else:
                if is_classifier:
                    model = LogisticRegression()
                else:
                    model = LinearRegression()

            model.fit(X, y)
            predicted = model.predict(X)
            predicted = predicted.ravel()

            # compute residuals
            if not is_classifier:
                residual = (y - predicted).ravel()
            else:
                residual = None

            train_result[target_name] = {
                "model": model,
                "parent_names": parent_names,
                "predicted": predicted,
                "residual": residual,
            }

        return train_result

    def _simulate(
        self,
        changing_exog,
        changing_models,
        impl,
        train_result,
        shuffle_residual,
        random_state,
    ):
        simulated = pd.DataFrame(index=np.arange(impl.exog_length_), columns=impl.endog_names_)

        if shuffle_residual:
            shuffle_index = random_state.choice(
                np.arange(impl.exog_length_),
                size=impl.exog_length_,
                replace=False
            )

        # modify causal order
        changing_edges = {}
        for target_name, info in changing_models.items():
            changing_edges[target_name] = info["parent_names"]
        causal_order = self._impl.get_causal_order(changing_edges)

        # predict from upstream to downstream
        for target_name in causal_order:
            is_classifier = target_name in impl.discrete_endog_names_

            # data
            parent_names = impl.get_parent_names(target_name)
            if target_name in changing_models.keys():
                parent_names_ = changing_models[target_name]["parent_names"]
                if parent_names_ is not None:
                    parent_names = parent_names_

            # error
            if target_name not in changing_exog.keys():
                error = train_result[target_name]["residual"]
            else:
                error = changing_exog[target_name].ravel()

            if shuffle_residual and error is not None:
                error = error[shuffle_index]

            # exogenous variable
            if len(parent_names) == 0:
                if not is_classifier:
                    simulated[target_name] = error
                else:
                    simulated[target_name] = impl.get_data(target_name)
                continue

            # model
            if target_name not in changing_models.keys():
                model = train_result[target_name]["model"]
            else:
                model = changing_models[target_name]["model"]

            # predict
            predicted = model.predict(simulated[parent_names].values)

            predicted = predicted.ravel()
            if not is_classifier:
                predicted = predicted + error

            simulated[target_name] = predicted

        return simulated


class CBSImpl(metaclass=ABCMeta):
    """ Class for handling a given causal_graph and data
    """

    def __init__(self, X, causal_graph, is_discrete=None):
        """ Constructor

        Constructor must expose properties below.
        """
        raise NotImplementedError

    @property
    def endog_names_(self):
        """ List of endogenous variable names
        Names of variables must be unique.
        """
        raise NotImplementedError

    @property
    def discrete_endog_names_(self):
        """ List of discrete endogenous variable names
        """
        raise NotImplementedError

    @property
    def causal_order_(self):
        """ List showing the causal order in a given causal graph.
        """
        raise NotImplementedError

    @property
    def exog_length_(self):
        """ Length of a estiamted residual data.
        """
        raise NotImplementedError

    @abstractmethod
    def get_parent_names(self, var_name):
        """ Return parent names of var_name.

        Parameters
        ----------
        var_name : str
            Target variable name.

        Returns
        -------
        parent_names : list
            List contains parent names of var_name.
        """
        raise NotImplementedError

    @abstractmethod
    def get_data(self, var_names):
        """ Return data of var_names

        Parameters
        ----------
        var_names : list
            List of target variable names.

        Returns
        -------
        data : numpy.ndarray
            Data of var_names.
        """
        raise NotImplementedError

    @abstractmethod
    def get_causal_order(self, changing_edges=None):
        """ Return a causal order in a given causal_graph and changing_edges

        Parameters
        ----------
        changing_edges : dict, default=None
            Dictaionry about edge modifications. Key is a target endogenous variable name and
            value is a list of parent names of the target endogenous variable.

        Returns
        -------
        causal_order : list
            List of endogenous variable names.
        """
        raise NotImplementedError


class CBSILiNGAM(CBSImpl):
    """ Class for data handling for ICALiNGAM and DirectLiNGAM. """

    def __init__(self, X, causal_graph, is_discrete=None):
        X_ = check_array(X)

        n_samples, n_features = X_.shape

        causal_graph = self._check_causal_graph(causal_graph, X_)

        endog_names, discrete_endog_names = self._make_var_names(causal_graph, X, is_discrete)

        causal_order = self._calc_causal_order(causal_graph)
        if causal_order is None:
            raise ValueError("causal_graph must be acyclic.")
        causal_order = [endog_names[n] for n in causal_order]

        self._X = X_
        self._exog_length = n_samples
        self._is_discrete = is_discrete
        self._endog_names = endog_names
        self._discrete_endog_names = discrete_endog_names
        self._causal_graph = causal_graph
        self._causal_order = causal_order

    @property
    def causal_order_(self):
        return self._causal_order

    @property
    def endog_names_(self):
        return self._endog_names

    @property
    def discrete_endog_names_(self):
        return self._discrete_endog_names

    @property
    def exog_length_(self):
        return self._exog_length

    def get_parent_names(self, var_name):
        if var_name not in self._endog_names:
            raise ValueError(f"Unknown variable. {var_name}")

        causal_graph = ~np.isclose(self._causal_graph, 0)

        index = self._endog_names.index(var_name)
        parent_indices = np.argwhere(causal_graph[index, :]).ravel()
        parent_names = np.array(self._endog_names)[parent_indices].tolist()

        return parent_names

    def get_data(self, var_names):
        if isinstance(var_names, str):
            var_names = [var_names]

        var_indices = []
        for var_name in var_names:
            index = self._endog_names.index(var_name)
            var_indices.append(index)
        data = self._X[:, var_indices]

        return data

    def get_causal_order(self, changing_edges=None):
        if changing_edges is None:
            changing_edges = {}

        causal_graph = self._causal_graph.copy()

        for target_name, parent_names in changing_edges.items():
            row = self._endog_names.index(target_name)
            cols = [self._endog_names.index(parent_name) for parent_name in parent_names]

            causal_graph[row, :] = 0
            causal_graph[row, cols] = 1

        causal_order = self._calc_causal_order(causal_graph)
        if causal_order is None:
            mes = "causal_graph updated by changing_models is cyclic."
            mes += " changing_models must be set so that causal graph does not cycle."
            raise ValueError(mes)
        causal_order = [self._endog_names[n] for n in causal_order]

        return causal_order

    def _check_causal_graph(self, causal_graph, X):
        try:
            causal_graph = check_array(causal_graph, copy=True)

            n_features = X.shape[1]
            if causal_graph.shape != (n_features, n_features):
                raise ValueError("The shape of causal_graph must be (n_features, n_features)")
        except Exception as e:
            raise ValueError("causal_graph has an error: " + str(e))

        return causal_graph

    def _make_var_names(self, causal_graph, X, is_discrete):
        n_features = len(causal_graph)

        if isinstance(X, pd.DataFrame):
            endog_names = check_array(X.columns, ensure_2d=False, dtype=None, copy=True)
            endog_names = [str(endog_name) for endog_name in endog_names]
        else:
            endog_names = [f"{i:d}" for i in range(n_features)]

        if is_discrete is None:
            is_discrete = [False for _ in range(n_features)]

        discrete_endog_names = np.array(endog_names)[is_discrete].tolist()

        return endog_names, discrete_endog_names

    def _calc_causal_order(self, causal_graph):
        """Obtain a causal order from the given causal_graph strictly.

        Parameters
        ----------
        causal_graph : array-like, shape (n_features, n_samples)
            Target causal_graph.

        Return
        ------
        causal_order : array, shape [n_features, ]
            A causal order of the given causal_graph on success, None otherwise.
        """
        causal_order = []

        row_num = causal_graph.shape[0]
        original_index = np.arange(row_num)

        while 0 < len(causal_graph):
            # find a row all of which elements are zero
            row_index_list = np.where(np.sum(np.abs(causal_graph), axis=1) == 0)[0]
            if len(row_index_list) == 0:
                break

            target_index = row_index_list[0]

            # append i to the end of the list
            causal_order.append(original_index[target_index])
            original_index = np.delete(original_index, target_index, axis=0)

            # remove the i-th row and the i-th column from causal_graph
            mask = np.delete(np.arange(len(causal_graph)), target_index, axis=0)
            causal_graph = causal_graph[mask][:, mask]

        if len(causal_order) != row_num:
            causal_order = None

        return causal_order


class CBSIUnobsCommonCauseLiNGAM(CBSILiNGAM):
    """ Class for data handling for BottomUpParceLiNGAM. """

    def _check_causal_graph(self, causal_graph, X):
        try:
            # fill nan with zeros
            causal_graph = check_array(causal_graph, force_all_finite="allow-nan", copy=True)

            n_features = X.shape[1]
            if causal_graph.shape != (n_features, n_features):
                raise RuntimeError("The shape of causal_graph must be (n_features, n_features)")

            causal_graph[np.isnan(causal_graph)] = 0
        except Exception as e:
            raise ValueError("causal_graph has an error: " + str(e))

        return causal_graph


class CBSITimeSeriesLiNGAM(CBSILiNGAM):
    """ Class for data handling for VARLiNGAM. """

    def __init__(self, X, causal_graph, is_discrete=None):
        super().__init__(X, causal_graph, is_discrete=is_discrete)

        # data is shortened to make lags
        self._exog_length = self._exog_length - self._n_lags

    def get_data(self, var_names):
        n_features = self._X.shape[1]

        if isinstance(var_names, str):
            var_names = [var_names]

        X_ = []
        for var_name in var_names:
            index = self._endog_names.index(var_name)
            X_index = index % n_features
            lag = index // n_features

            data = self._X[self._n_lags - lag:-lag if lag != 0 else None, X_index]

            X_.append(data)
        X_ = np.vstack(X_).T

        return X_

    def _check_causal_graph(self, causal_graph, X):
        try:
            causal_graph = check_array(causal_graph, allow_nd=True, copy=True)
            if len(causal_graph.shape) != 3:
                raise ValueError("causal_graph must be 3 dimentional array.")

            n_features = X.shape[1]
            if causal_graph.shape[1:] != (n_features, n_features):
                raise ValueError("The shape of causal_graph must be (n_lags+1, n_features, n_features)")
        except Exception as e:
            raise ValueError("causal_graph has an error: " + str(e))

        coef = np.concatenate(causal_graph, axis=1)

        causal_graph_ = np.zeros((coef.shape[1], coef.shape[1]))
        causal_graph_[:len(coef)] = coef

        self._n_lags = len(causal_graph) - 1

        return causal_graph_

    def _make_var_names(self, causal_graph, X, is_discrete):
        n_features = X.shape[1]

        if isinstance(X, pd.DataFrame):
            endog_names_ = X.columns.tolist()
        else:
            endog_names_ = [f"{i:d}" for i in range(n_features)]

        endog_names = []
        for i in range(self._n_lags + 1):
            if i == 0:
                index_format = "[t]"
            else:
                index_format = f"[t-{i}]"
            endog_names += [name + index_format for name in endog_names_]

        if is_discrete is None:
            is_discrete = [False for _ in range(n_features)]
            is_discrete *= self._n_lags + 1
        else:
            is_discrete = is_discrete * (self._n_lags + 1)

        discrete_endog_names = np.array(endog_names)[is_discrete].tolist()

        return endog_names, discrete_endog_names


class _LinearRegression():
    """ Linear regression model with configurable coefficients """

    def __init__(self, coef):
        self._coef = np.array(coef)

    def predict(self, X):
        if len(self._coef) == 0:
            return np.zeros((len(X), 1))
        return (self._coef @ X.T).T

    @property
    def coef_(self):
        return self._coef


class _LogisticRegression():
    """ Logistic regression model with configurable coefficients """

    def __init__(self, coef):
        self._coef = np.array(coef)

    def predict(self, X):
        if len(self._coef) == 0:
            return np.zeros((len(X), 1))
        return expit((self._coef @ X.T).T) >= 0.5

    @property
    def coef_(self):
        return self._coef
