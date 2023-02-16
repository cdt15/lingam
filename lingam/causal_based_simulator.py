from collections import namedtuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils import check_array
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

TrainData = namedtuple("TrainData", ("condition", "X", "y"))
TrainResult = namedtuple("TrainResult", ("condition", "model", "exp_columns", "predicted", "residual"))
ChangingModels = namedtuple("ChangingModels", ("name", "condition", "model"))


class CausalBasedSimulator:
    """
    Causal based simulator.

    Attributes
    ----------
    train_result_ : dict of string -> list of dict
        information about trained models.
    residual_ : pandas.DataFrame
        residuals of trained models.
    categorical_info_ : dict of strig -> list of str
        information about categorical variables.
        the key of its dictioary is variable name, the value is the list of classes.
    simulated_data_ : pandas.DataFrame
        result of simulation.
    """

    def __init__(self):
        # input of train()
        self._data = None
        self._causal_graph = None

        # result of train()
        self._train_results = None

        # input of run()
        self._changing_exog = None
        self._changing_models = None

        # result of run()
        self._causal_order = None
        self._simulated_result = None

    def train(self, X, causal_graph, models=None):
        """
        Estimate functional relations between variables and variable distributions based on
        the training data ``X`` and the causal graph ``G``. The functional relations represents
        by sklearn.linear_model.LinearRegression if the object variable is numeric,
        and represents by sklearn.linear_model.LogisticRegression if the object variable is categorical by default.
        ``train_result_`` and ``residual_`` will be exposed after executing train().

        Parameters
        ----------
        X : pandas.DataFrame
            Training data.
        causal_graph : array-like of shape (n_features, _features)
            Causal graph.
        models : dict of string -> object, default=None
            Dictionary about models of variables.
            Models are cloned internaly and are trained to infer functioal relations.
            Given instances of the model are cloned to estimate the functional relation between variables.

        Returns
        -------
        self : Object
        """

        data, categorical_info = self._check_data(X)

        causal_graph = self._check_causal_graph(causal_graph, data.shape[1])
        train_models = self._check_models(models, data.index, data.columns, categorical_info)

        train_results = self._train(data, categorical_info, causal_graph, train_models)

        residual_df = {}
        for column in X.columns:
            if len(train_results[column]) > 0:
                series = self._concat_residuals(train_results, column, data.index)
            else:
                series = np.ones(data.shape[0]) * np.nan
            residual_df[column] = series
        residual_df = pd.DataFrame(residual_df)

        train_result = self._prepare_to_expose_train(train_results, categorical_info)

        self.train_result_ = train_result
        self.residual_ = residual_df

        self._data = data
        self.categorical_info_ = categorical_info
        self._train_results = train_results
        self._causal_graph = causal_graph

        return self

    def run(self, changing_exog=None, changing_models=None, shuffle_residual=False, random_state=None):
        """
        Generate simulated data using trained models and the given causal graph with given exogenous data and models.
        Specifying environmental changes to ``changing_exog`` or specifiyig changes in fucitonal relation to ``change_models``
        effects simulated data.
        Residuals to simulate variables are shuffled using radom_state if ``shuffle_residual`` is True.
        ``simulated_data_`` will be expose after excuting train().

        Parameters
        ----------
        changing_exog : dict of string -> array-like, default=None
            Dictioary about exogeous variables which keys are variable names and values are data of variables.
            That variable name should be a one of column names of X and the length should be same as X.
        changing_model : list of dict, default=None
            List of the changing models which elements are dictionary. that keys should be name, condition, and model.
            For name and condition, refer to ``train_result_`` and set the values corresponding to
            the conditions you wish to change. For model, you must provide a trained machine learning instance.
        shuffle_residual : bool, default=True
            If True, residuals are shuffled.
        random_state : int, default=None
            If shuffle_residual is True, random_state is used as seed.

        Returns
        -------
        simulated_data : pandas.DataFrame
            simulated data.
        """

        if self._train_results is None:
            raise RuntimeError("train() should be executed before run() is executed.")

        changing_exog_df = self._check_changing_exog(
            changing_exog, self._data.index, self._data.columns, self.categorical_info_)
        changing_models2 = self._check_changing_models(changing_models, self.categorical_info_, self._train_results)

        if self._causal_order is not None:
            causal_order = self._causal_order
        else:
            causal_order = self._get_causal_order(self._causal_graph)

        simulated_df = self._simulate(self._data, self._causal_graph, causal_order, self.categorical_info_,
                                      changing_exog_df, changing_models2, self._train_results, shuffle_residual, random_state)

        self.simulated_data_ = simulated_df

        self._changing_exog = changing_exog_df
        self._changing_models = changing_models2
        if self._causal_order is None:
            self._causal_order = causal_order

        return self.simulated_data_

    def _check_data(self, X):
        if not isinstance(X, pd.DataFrame):
            raise RuntimeError("X must be a pandas.DataFrame.")

        # copy to replace values of categorical variables.
        X = X.copy()

        is_categorical = {c: is_categorical for c, is_categorical in zip(X.columns, X.dtypes.values == "category")}

        categorical_info = {}
        for column, is_categorical_ in is_categorical.items():
            if not is_categorical_:
                continue

            # replace values of the categorical variable
            categorical_info[column] = np.unique(X[column]).tolist()
            X[column] = X[column].apply(lambda x: categorical_info[column].index(x))

        return X, categorical_info

    def _check_causal_graph(self, causal_graph, column_num):
        causal_graph_ = check_array(causal_graph)
        if causal_graph_.shape[0] != column_num or causal_graph_.shape[1] != column_num:
            raise RuntimeError("The height and width of the causal_graph must be the same length.")
        return causal_graph_

    def _check_model_instance(self, model, column, categorical_info):
        model_type = RegressorMixin if column not in categorical_info.keys() else ClassifierMixin
        if isinstance(model, Pipeline):
            if not isinstance(model.steps[-1][-1], model_type):
                raise RuntimeError(
                    "The final step in the Pipeline should be an instance of the regression/classification model.")
        elif isinstance(model, BaseSearchCV):
            if not isinstance(model.get_params()["estimator"], model_type):
                raise RuntimeError("Must be an instance of a regression/classification model.")
        else:
            if not isinstance(model, model_type):
                raise RuntimeError("Must be an instance of a regression/classification model.")

        if model_type == ClassifierMixin:
            try:
                func = getattr(model, "predict_proba")
                if not callable(func):
                    raise Exception
            except Exception:
                raise RuntimeError("Classification models without predict_proba() cannot be used.")

    def _check_models(self, models, index, columns, categorical_info):
        if models is None:
            return {}

        if not isinstance(models, dict):
            raise RuntimeError("models must be a dictionary.")

        for column, model in models.items():
            if column not in columns:
                raise RuntimeError("The key of models must be a variable name that exists in X.")

            self._check_model_instance(model, column, categorical_info)

        return models

    def _check_changing_exog(self, changing_exog, index, columns, categorical_info):
        if changing_exog is None:
            return pd.DataFrame()

        if not isinstance(changing_exog, dict):
            raise RuntimeError("changing_exog must be a dictionary.")

        changing_exog_df = {}
        for column, values in changing_exog.items():
            if column not in columns:
                raise RuntimeError("The key of changing_exog must be a variable name that exists in X.")

            if column in categorical_info.keys():
                raise RuntimeError("You cannot set changig_exog for a column of category variables.")

            s = check_array(values, ensure_2d=False, dtype=None).flatten()
            if s.shape[0] != len(index):
                raise RuntimeError("The data sequence of changing_exog values must be the same length as X.")

            changing_exog_df[column] = pd.Series(values, index=index)
        changing_exog_df = pd.DataFrame(changing_exog_df).loc[index, :]

        return changing_exog_df

    def _check_changing_models(self, changing_models, categorical_info, train_results):
        if changing_models is None:
            return {}

        if not isinstance(changing_models, list):
            raise RuntimeError("changing_models must be a list.")

        changing_models_ = []
        for model_info in changing_models:
            if not isinstance(model_info, dict):
                raise RuntimeError("The elements of changing_models must be dict.")

            if len(set(ChangingModels._fields) - set(model_info.keys())) > 0:
                raise RuntimeError("The models element is missing a key.")

            name = model_info["name"]
            if name not in train_results.keys():
                raise RuntimeError("A variable name that does not exist.")

            condition = model_info["condition"]
            conditions = [cond_result.condition for cond_result in train_results[name]]
            if condition not in conditions:
                raise RuntimeError("This is a condition that does not exist in the training results.")

            model = model_info["model"]
            self._check_model_instance(model, name, categorical_info)

            changing_models_.append(ChangingModels(name=name, condition=condition, model=model))
        return changing_models_

    def _get_causal_order(self, causal_graph):
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

            # remove the i-th row and the i-th column from matrix
            mask = np.delete(np.arange(len(causal_graph)), target_index, axis=0)
            causal_graph = causal_graph[mask][:, mask]

        if len(causal_order) != row_num:
            causal_order = None

        return np.array(causal_order)

    def _get_train_data(self, data, X_columns, y_column, categorical_info, changing_exog_df=None):
        train_data_list = []

        # column names of categorical varialbes in X_columns
        X_columns_cat = set(X_columns) & set(categorical_info.keys())
        X_columns_cat = sorted(X_columns_cat, key=lambda x: list(X_columns).index(x))

        if len(X_columns_cat) == 0:
            X = data.loc[:, X_columns].copy()
            y = data.loc[:, y_column].copy()

            # apply exogeous
            if changing_exog_df is not None:
                for col, series in changing_exog_df.items():
                    if col in X.columns.tolist():
                        X[col] = series

            train_data_list.append(TrainData(condition=None, X=X, y=y))
            return train_data_list

        # column names of non-ategorical variables
        X_columns_num = set(X_columns) - set(categorical_info.keys())
        X_columns_num = sorted(X_columns_num, key=lambda x: list(X_columns).index(x))

        # unique conditions by parent nodes
        uniq_conds = np.unique(data.loc[:, X_columns_cat].values.tolist(), axis=0)
        for uniq_cond in uniq_conds:
            # filter X by the condition
            target_index = np.argwhere(np.all(data.loc[:, X_columns_cat].values == uniq_cond, axis=1)).flatten()
            X_filtered = data.iloc[target_index, :].loc[:, X_columns_num].copy()
            y_filtered = data.iloc[target_index, :].loc[:, y_column].copy()

            # apply exogeous
            if changing_exog_df is not None:
                for col, series in changing_exog_df.items():
                    if col in X.columns.tolist():
                        X_filtered[col] = series.loc[target_index]

            condition = {}
            for column, value in zip(X_columns_cat, uniq_cond):
                condition[column] = categorical_info[column][value]

            train_data_list.append(TrainData(condition=condition, X=X_filtered, y=y_filtered))

        return train_data_list

    def _predict_reg_model(self, X, model):
        try:
            predicted = model.predict(X.values)
        except Exception as e:
            raise RuntimeError("An exception occurred during predict() of the model.:" + str(e))

        return np.array(predicted)

    def _train_reg_model(self, X, y, model):
        try:
            model.fit(X.values, y.values)
        except Exception as e:
            raise RuntimeError("An exception occurred during fit() of the model.:" + str(e))

        predicted = self._predict_reg_model(X, model)
        predicted = pd.Series(predicted, index=X.index, dtype=y.dtype)

        resid = y - predicted

        return model, predicted, resid

    def _predict_clf_model(self, X, model):
        try:
            proba = model.predict_proba(X.values)
        except Exception as e:
            raise RuntimeError("An exception occurred during predict_proba() of the model.:" + str(e))

        # sampling values based on predicted probability
        predicted = []
        for proba_ in proba:
            predicted.append(np.random.choice(model.classes_, p=proba_))

        return np.array(predicted)

    def _train_clf_model(self, X, y, model, categorical_info):
        try:
            model.fit(X.values, y.values)
        except Exception as e:
            raise RuntimeError("An exception occurred during fit() of the model.:" + str(e))

        predicted = self._predict_clf_model(X, model)
        predicted = pd.Series(predicted, index=X.index, dtype=y.dtype)

        # classifier doesn't have residuals
        resid = None

        return model, predicted, resid

    def _concat_residuals(self, train_results, to_name, index):
        ret = []

        cond_results = train_results[to_name]
        for cond_result in cond_results:
            series = getattr(cond_result, "residual")
            if series is None:
                predicted = getattr(cond_result, "predicted")
                series = pd.Series(np.ones(predicted.shape[0]) * np.nan, index=predicted.index)
            ret.append(series)
        ret = pd.concat(ret, axis=0).loc[index]

        return ret

    def _concat_predicts(self, train_results, to_name, index):
        ret = []

        cond_results = train_results[to_name]
        for cond_result in cond_results:
            series = getattr(cond_result, "predicted")
            ret.append(series)
        ret = pd.concat(ret, axis=0).loc[index]

        return ret

    def _prepare_to_expose_train(self, train_results, categorical_info):
        train_results_ = {}
        for to_name, cond_results in train_results.items():
            train_result_ = []
            for result in cond_results:
                result_ = {}
                if result.condition is None:
                    result_["condition"] = None
                else:
                    result_["condition"] = {k: v for k, v in result.condition.items()}
                result_["model"] = result.model
                result_["exp_columns"] = result.exp_columns

                if to_name not in categorical_info.keys():
                    # non categorical
                    result_["predicted"] = result.predicted
                else:
                    # categorical
                    result_["predicted"] = result.predicted.map(lambda x: categorical_info[to_name][x])

                result_["residual"] = result.residual

                train_result_.append(result_)
            train_results_[to_name] = train_result_
        return train_results_

    def _train(self, data, categorical_info, causal_graph, models):
        train_results = {}

        for to_index, graph_row in enumerate(causal_graph):
            cond_results = []

            to_name = data.columns[to_index]

            from_indices = np.argwhere(~np.isclose(graph_row, 0)).flatten()
            from_names = data.columns[from_indices]

            # making data list splitted by conditions
            train_data_list = self._get_train_data(data, from_names, to_name, categorical_info)

            from_names_num = sorted(set(from_names) - set(categorical_info.keys()), key=lambda x: list(data.columns).index(x))

            # most topstream
            if len(from_names) == 0:
                train_results[to_name] = []
                continue

            for train_data in train_data_list:
                if to_name in models.keys():
                    # given model
                    model = clone(models[to_name])
                else:
                    # default model
                    if len(from_names_num) > 0:
                        if to_name in categorical_info.keys():
                            model = LogisticRegression()
                        else:
                            model = LinearRegression()
                    else:
                        # no explanatory (only categorical variables)
                        if to_name in categorical_info.keys():
                            model = CbsCategoricalClassifier()
                        else:
                            model = CbsExpectedValueRegressor()

                if to_name in categorical_info.keys():
                    model, predicted, resid = self._train_clf_model(train_data.X, train_data.y, model, categorical_info)
                else:
                    model, predicted, resid = self._train_reg_model(train_data.X, train_data.y, model)

                cond_results.append(TrainResult(condition=train_data.condition,
                                                model=model, exp_columns=from_names_num, predicted=predicted, residual=resid))
            train_results[to_name] = cond_results
        return train_results

    def _select_changing_model(self, changing_models, to_name, condition):
        for model_info in changing_models:
            if model_info.name != to_name:
                continue
            elif model_info.condition != condition:
                continue
            return model_info.model
        return None

    def _simulate(self, data, causal_graph, causal_order, categorical_info, changing_exog_df,
                  changing_models, train_results, shuffle_residual, random_state):
        sim_df = pd.DataFrame(index=data.index)

        # predict from upstream to downstream
        for to_index in causal_order:
            # input column of node i is the nonzero column of row i of the adjacency matrix.
            from_indices = causal_graph[to_index, :]
            from_indices = np.argwhere(~np.isclose(from_indices, 0)).flatten()

            to_name = data.columns[to_index]
            from_names = [data.columns[from_index] for from_index in from_indices]

            # no predictions by model for root node. The data is same as X.
            if len(from_names) == 0:
                sim_df[to_name] = data[to_name].copy()
                continue

            if to_name in changing_exog_df.columns:
                error = changing_exog_df[to_name]
            elif to_name not in categorical_info.keys():
                error = self._concat_residuals(train_results, to_name, data.index)

            X = sim_df[from_names]

            # simulate for each conditions
            y_sim = []
            for cond_result in train_results[to_name]:
                X_ = X

                if cond_result.condition is not None:
                    # filter rows by condition
                    filter_ = []
                    for key, value in cond_result.condition.items():
                        filter_.append(X_[key] == categorical_info[key].index(value))
                    filter_ = np.all(filter_, axis=0)
                    X_ = X_.loc[filter_, :]

                # filter columns by explanatory
                if len(cond_result.exp_columns) > 0:
                    X_ = X_.loc[:, cond_result.exp_columns]
                else:
                    # dummy data when all parent nodes are categorical
                    X_ = pd.DataFrame(np.ones(X_.shape[0]).reshape(X_.shape[0]) * np.nan, index=X_.index, columns=[to_name])

                # select a model to simulate
                model = self._select_changing_model(changing_models, to_name, cond_result.condition)
                if model is None:
                    model = cond_result.model

                if to_name in categorical_info.keys():
                    y_hat = self._predict_clf_model(X_, model)
                else:
                    y_hat = self._predict_reg_model(X_, model)

                    cond_error = error.loc[X_.index]
                    if shuffle_residual is True:
                        cond_error = cond_error.sample(frac=1, random_state=random_state)
                    y_hat += cond_error

                y_sim.append(pd.Series(y_hat, index=X_.index))
            sim_df[to_name] = pd.concat(y_sim, axis=0).loc[data.index]
        sim_df = sim_df.loc[data.index, data.columns]

        # decode categorical values
        for column, categories in categorical_info.items():
            sim_df[column] = sim_df[column].map(lambda idx: categories[idx])
            sim_df[column] = sim_df[column].astype("category")

        return sim_df


class CbsExpectedValueRegressor(BaseEstimator, RegressorMixin):
    # Regression model to be used when there is no explanatory variable.

    def fit(self, X, y):
        self.expected_value_ = np.mean(y)

    def predict(self, X):
        y_hat = self.expected_value_ * np.ones(X.shape[0])
        return y_hat


class CbsCategoricalClassifier(BaseEstimator, ClassifierMixin):
    # classification model to be used when there is no explanatory variable.

    def fit(self, X, y):
        values, counts = np.unique(y, return_counts=True)
        p = [c / y.shape[0] for c in counts]

        self.p_ = p
        self.classes_ = values
        return self

    def predict_proba(self, X):
        return [self.p_] * X.shape[0]
