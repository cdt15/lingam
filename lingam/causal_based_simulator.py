from collections import namedtuple

import numpy as np
import pandas as pd

from sklearn.base import RegressorMixin, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils import check_array
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

TrainData = namedtuple("TrainData", ("condition", "X", "y"))
TrainResult = namedtuple(
    "TrainResult", ("condition", "model", "exp_columns", "predicted", "residual")
)
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

    cat_map_ : dict of strig -> list of str
        information about categorical variables. the key of its dictioary is
        variable name, the value is the list of classes.

    simulated_data_ : pandas.DataFrame
        result of simulation.
    """

    def train(self, X, causal_graph, models=None):
        """
        Estimate functional relations between variables and variable
        distributions based on the training data ``X`` and the causal graph
        ``G``. The functional relations represents by
        sklearn.linear_model.LinearRegression if the object variable is
        numeric, and represents by sklearn.linear_model.LogisticRegression
        if the object variable is categorical by default. ``train_result_``
        and ``residual_`` will be exposed after executing train().

        Parameters
        ----------
        X : pandas.DataFrame
            Training data.

        causal_graph : array-like of shape (n_features, _features)
            Causal graph.

        models : dict of string -> object, default=None
            Dictionary about models of variables. Models are cloned internaly
            and are trained to infer functioal relations. Given instances of
            the model are cloned to estimate the functional relation between
            variables.

        Returns
        -------
        self : Object
        """

        # checking inputs
        data, cat_map = self._check_data(X)

        causal_graph = self._check_causal_graph(causal_graph, data.shape[1])

        train_models = self._check_models(models, data.index, data.columns, cat_map)

        # training
        train_results = self._train(data, cat_map, causal_graph, train_models)

        residual_df = {}
        for col_name in X.columns:
            if len(train_results[col_name]) > 0:
                series = self._concat_residuals(train_results, col_name, data.index)
            else:
                # node with no parent nodes
                series = np.ones(data.shape[0]) * np.nan

            residual_df[col_name] = series

        residual_df = pd.DataFrame(residual_df)

        # setting attributes
        self.train_result_ = self._prep_to_expose_train(train_results, cat_map)
        self.residual_ = residual_df
        self.categorical_info_ = cat_map

        # setting information
        self._data = data
        self._train_results = train_results
        self._causal_graph = causal_graph
        self._causal_order = None

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
        if ``shuffle_residual`` is True. ``simulated_data_`` will be
        expose after excuting train().

        Parameters
        ----------
        changing_exog : dict of string -> array-like, default=None
            Dictioary about exogeous variables which keys are variable
            names and values are data of variables. That variable name
            should be a one of column names of X and the length should
            be same as X.

        changing_model : list of dict, default=None
            List of the changing models which elements are dictionary. that
            keys should be name, condition, and model. For name and
            condition, refer to ``train_result_`` and set the values
            corresponding to the conditions you wish to change. For model,
            you must provide a trained machine learning instance.

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
            raise RuntimeError("run() shall be executed after train() is executed")

        # checking inputs
        changing_exog_df = self._check_changing_exog(
            changing_exog, self._data.index, self._data.columns, self.categorical_info_
        )
        changing_models2 = self._check_changing_models(
            changing_models, self.categorical_info_, self._train_results
        )

        # calculating causal_order if it has not been calculated
        if self._causal_order is not None:
            causal_order = self._causal_order
        else:
            causal_order = self._get_causal_order(self._causal_graph)

        simulated_df = self._simulate(
            self._data,
            self._causal_graph,
            causal_order,
            self.categorical_info_,
            changing_exog_df,
            changing_models2,
            self._train_results,
            shuffle_residual,
            random_state,
        )

        # set attributes
        self.simulated_data_ = simulated_df

        # set information
        self._changing_exog = changing_exog_df
        self._changing_models = changing_models2
        if self._causal_order is None:
            self._causal_order = causal_order

        return self.simulated_data_

    def _check_data(self, X):
        if not isinstance(X, pd.DataFrame):
            raise RuntimeError("X shall be a pandas.DataFrame.")

        cat_map = {}
        for c, is_cat in zip(X.columns, X.dtypes.values == "category"):
            if not is_cat:
                continue

            cat_map[c] = np.unique(X[c]).tolist()
            X[c] = X[c].apply(lambda x: cat_map[c].index(x))

        return X, cat_map

    def _check_causal_graph(self, causal_graph, col_num):
        graph = check_array(causal_graph)

        if graph.shape[0] != col_num or graph.shape[1] != col_num:
            raise RuntimeError("causal_graph.shape shall be square.")

        return graph

    def _check_model_instance(self, model, col_name, cat_map):
        if col_name not in cat_map.keys():
            model_type = RegressorMixin
        else:
            model_type = ClassifierMixin

        if isinstance(model, Pipeline):
            if not isinstance(model.steps[-1][-1], model_type):
                raise RuntimeError(
                    "The last step in Pipeline should be an "
                    + "instance of a regression/classification model."
                )
        elif isinstance(model, BaseSearchCV):
            if not isinstance(model.get_params()["estimator"], model_type):
                raise RuntimeError(
                    "The type of the estimator shall be an "
                    + "instance of a regression/classification model."
                )
        else:
            if not isinstance(model, model_type):
                raise RuntimeError(
                    "The type of the estimator shall be an "
                    + "instance of a regression/classification model."
                )

        if model_type == ClassifierMixin:
            try:
                func = getattr(model, "predict_proba")
                if not callable(func):
                    raise Exception
            except Exception:
                raise RuntimeError(
                    "Classification models shall have " + "predict_proba()."
                )

    def _check_models(self, models, index, columns, cat_map):
        if models is None:
            return {}

        if not isinstance(models, dict):
            raise RuntimeError("models shall be a dictionary.")

        for col_name, model in models.items():
            if col_name not in columns:
                raise RuntimeError(f"Unknown column name ({col_name})")

            self._check_model_instance(model, col_name, cat_map)

        return models

    def _check_changing_exog(self, changing_exog, index, columns, cat_map):
        if changing_exog is None:
            return pd.DataFrame()

        if not isinstance(changing_exog, dict):
            raise RuntimeError("changing_exog shall be a dictionary.")

        changing_exog_df = {}
        for col_name, values in changing_exog.items():
            if col_name not in columns:
                raise RuntimeError(f"Unknown key in changing_exog. ({col_name})")

            if col_name in cat_map.keys():
                raise RuntimeError(
                    f"Category variables shall not be specified. ({col_name})"
                )

            s = check_array(values, ensure_2d=False, dtype=None).flatten()
            if s.shape[0] != len(index):
                raise RuntimeError(f"Wrong length. ({s.shape[0]} != {len(index)})")

            changing_exog_df[col_name] = pd.Series(values, index=index)
        changing_exog_df = pd.DataFrame(changing_exog_df).loc[index, :]

        return changing_exog_df

    def _check_changing_models(self, changing_models, cat_map, train_results):
        if changing_models is None:
            return {}

        if not isinstance(changing_models, list):
            raise RuntimeError("changing_models shall be a list.")

        changing_models_ = []
        for model_info in changing_models:
            if not isinstance(model_info, dict):
                raise RuntimeError("changing_models shall be a list of dictionaries.")

            missing_keys = set(ChangingModels._fields) - set(model_info.keys())
            if len(missing_keys) > 0:
                raise RuntimeError("Missing key on model_info. " + str(missing_keys))

            name = model_info["name"]
            if name not in train_results.keys():
                raise RuntimeError(f"Unknown name. ({name})")

            condition = model_info["condition"]
            conditions = [cond_result.condition for cond_result in train_results[name]]
            if condition not in conditions:
                raise RuntimeError("Not-exsitent consition. " + str(condition))

            model = model_info["model"]
            self._check_model_instance(model, name, cat_map)

            changing_models_.append(
                ChangingModels(name=name, condition=condition, model=model)
            )
        return changing_models_

    def _get_causal_order(self, causal_graph):
        causal_order = []

        row_num = causal_graph.shape[0]
        original_index = np.arange(row_num)

        while 0 < len(causal_graph):
            # finding rows where the elements are all zeros
            row_index_list = np.where(np.sum(np.abs(causal_graph), axis=1) == 0)[0]
            if len(row_index_list) == 0:
                break

            target_index = row_index_list[0]

            # append i to the end of the list
            causal_order.append(original_index[target_index])
            original_index = np.delete(original_index, target_index, axis=0)

            # remove i-th row and i-th column from matrix
            mask = np.delete(np.arange(len(causal_graph)), target_index, axis=0)
            causal_graph = causal_graph[mask][:, mask]

        if len(causal_order) != row_num:
            causal_order = None

        return np.array(causal_order)

    def _get_train_data(self, data, X_cols, y_col, cat_map, changing_exog_df=None):
        train_data_list = []

        # column names of categorical variables in X_columns
        X_cols_cat = set(X_cols) & set(cat_map.keys())
        X_cols_cat = sorted(X_cols_cat, key=lambda x: list(X_cols).index(x))

        # no categorical variable in X
        if len(X_cols_cat) == 0:
            X = data.loc[:, X_cols].copy()
            y = data.loc[:, y_col].copy()

            # overwrite with exogenous
            if changing_exog_df is not None:
                for col, series in changing_exog_df.items():
                    if col in X.columns.tolist():
                        X[col] = series

            train_data_list.append(TrainData(condition=None, X=X, y=y))
            return train_data_list

        # column names of non-categorical variables in X_columns
        X_cols_num = set(X_cols) - set(cat_map.keys())
        X_cols_num = sorted(X_cols_num, key=lambda x: list(X_cols).index(x))

        # unique conditions by categorical parent nodes
        uniq_conds = np.unique(data.loc[:, X_cols_cat].values.tolist(), axis=0)

        for uniq_cond in uniq_conds:
            # filter by the condition
            target_index = np.argwhere(
                np.all(data.loc[:, X_cols_cat].values == uniq_cond, axis=1)
            ).flatten()

            X_filtered = data.iloc[target_index, :].loc[:, X_cols_num].copy()
            y_filtered = data.iloc[target_index, :].loc[:, y_col].copy()

            # overwrite with exogenous
            if changing_exog_df is not None:
                for col, series in changing_exog_df.items():
                    if col in X.columns.tolist():
                        X_filtered[col] = series.loc[target_index]

            condition = {}
            for col_name, value in zip(X_cols_cat, uniq_cond):
                condition[col_name] = cat_map[col_name][value]

            train_data_list.append(
                TrainData(condition=condition, X=X_filtered, y=y_filtered)
            )

        return train_data_list

    def _predict_reg_model(self, X, model):
        try:
            predicted = model.predict(X.values)
        except Exception as e:
            raise RuntimeError(
                "An exception occurred during predict() of the model. " + str(e)
            )

        return np.array(predicted)

    def _train_reg_model(self, X, y, model):
        try:
            model.fit(X.values, y.values)
        except Exception as e:
            raise RuntimeError(
                "An exception occurred during fit() of the model. " + str(e)
            )

        predicted = self._predict_reg_model(X, model)
        predicted = pd.Series(predicted, index=X.index, dtype=y.dtype)

        resid = y - predicted

        return model, predicted, resid

    def _predict_clf_model(self, X, model):
        try:
            proba = model.predict_proba(X.values)
        except Exception as e:
            raise RuntimeError(
                "An exception occurred during predict_proba() of the model. " + str(e)
            )

        # sampling values based on predicted probability
        predicted = []
        for proba_ in proba:
            predicted.append(np.random.choice(model.classes_, p=proba_))

        return np.array(predicted)

    def _train_clf_model(self, X, y, model, cat_map):
        try:
            model.fit(X.values, y.values)
        except Exception as e:
            raise RuntimeError(
                "An exception occurred during fit() of the model. " + str(e)
            )

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
                series = pd.Series(
                    np.ones(predicted.shape[0]) * np.nan, index=predicted.index
                )

            ret.append(series)

        # concat and sort by original index
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

    def _prep_to_expose_train(self, train_results, cat_map):
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

                if to_name not in cat_map.keys():
                    # non categorical
                    result_["predicted"] = result.predicted
                else:
                    # categorical
                    result_["predicted"] = result.predicted.map(
                        lambda x: cat_map[to_name][x]
                    )

                result_["residual"] = result.residual

                train_result_.append(result_)
            train_results_[to_name] = train_result_

        return train_results_

    def _train(self, data, cat_map, graph, models):
        train_results = {}

        for to_index, graph_row in enumerate(graph):
            cond_results = []

            from_indices = np.argwhere(~np.isclose(graph_row, 0)).flatten()

            # get column names
            to_name = data.columns[to_index]
            from_names = data.columns[from_indices]
            from_names_num = sorted(
                set(from_names) - set(cat_map.keys()),
                key=lambda x: list(data.columns).index(x),
            )

            # making data list splitted by conditions
            cond_tr_data = self._get_train_data(data, from_names, to_name, cat_map)

            # node with no parents
            if len(from_names) == 0:
                train_results[to_name] = []
                continue

            # node with parents
            for tr_data in cond_tr_data:
                # selecting a model to train
                if to_name in models.keys():
                    model = clone(models[to_name])
                else:
                    if len(from_names_num) > 0:
                        # default
                        if to_name in cat_map.keys():
                            model = LogisticRegression()
                        else:
                            model = LinearRegression()
                    else:
                        # no non-categorical parents
                        if to_name in cat_map.keys():
                            model = DummyClassifier(strategy="uniform")
                        else:
                            model = DummyRegressor()

                # train model
                if to_name in cat_map.keys():
                    model, predicted, resid = self._train_clf_model(
                        tr_data.X, tr_data.y, model, cat_map
                    )
                else:
                    model, predicted, resid = self._train_reg_model(
                        tr_data.X, tr_data.y, model
                    )

                cond_results.append(
                    TrainResult(
                        condition=tr_data.condition,
                        model=model,
                        exp_columns=from_names_num,
                        predicted=predicted,
                        residual=resid,
                    )
                )
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

    def _simulate(
        self,
        data,
        causal_graph,
        causal_order,
        cat_map,
        changing_exog_df,
        changing_models,
        train_results,
        shuffle_residual,
        random_state,
    ):
        sim_df = pd.DataFrame(index=data.index)

        # predicting from upstream to downstream
        for to_index in causal_order:
            # input nodes of node i are nonzero variables of i-th row of graph
            from_indices = causal_graph[to_index, :]
            from_indices = np.argwhere(~np.isclose(from_indices, 0)).flatten()

            to_name = data.columns[to_index]
            from_names = [data.columns[from_index] for from_index in from_indices]

            # assigning values without predicting if to_name has no parents.
            if len(from_names) == 0:
                sim_df[to_name] = data[to_name].copy()
                continue

            if to_name in changing_exog_df.columns:
                error = changing_exog_df[to_name]
            elif to_name not in cat_map.keys():
                error = self._concat_residuals(train_results, to_name, data.index)

            X = sim_df[from_names]

            # simulation for each conditions
            y_sim = []
            for cond_result in train_results[to_name]:
                X_ = X

                if cond_result.condition is not None:
                    # filter rows by a condition
                    filter_ = []
                    for key, value in cond_result.condition.items():
                        filter_.append(X_[key] == cat_map[key].index(value))
                    filter_ = np.all(filter_, axis=0)
                    X_ = X_.loc[filter_, :]

                # filter columns by explanatories
                if len(cond_result.exp_columns) > 0:
                    X_ = X_.loc[:, cond_result.exp_columns]
                else:
                    # dummy data when all parent nodes are categorical
                    X_ = pd.DataFrame(
                        np.ones(X_.shape[0]).reshape(X_.shape[0]) * np.nan,
                        index=X_.index,
                        columns=[to_name],
                    )

                # selecting a model for the simulation
                model = self._select_changing_model(
                    changing_models, to_name, cond_result.condition
                )
                if model is None:
                    model = cond_result.model

                if to_name in cat_map.keys():
                    y_hat = self._predict_clf_model(X_, model)
                else:
                    y_hat = self._predict_reg_model(X_, model)

                    cond_error = error.loc[X_.index]
                    if shuffle_residual is True:
                        cond_error = cond_error.sample(
                            frac=1, random_state=random_state
                        )
                    y_hat += cond_error

                y_sim.append(pd.Series(y_hat, index=X_.index))
            sim_df[to_name] = pd.concat(y_sim, axis=0).loc[data.index]
        sim_df = sim_df.loc[data.index, data.columns]

        # decoding categorical values
        for col_name, categories in cat_map.items():
            sim_df[col_name] = sim_df[col_name].map(lambda idx: categories[idx])
            sim_df[col_name] = sim_df[col_name].astype("category")

        return sim_df
