import numpy as np

from sklearn.utils import check_scalar, check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import check_cv
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor


class OutOfSampleCausalTuning():
    """Implementation of Out-of-sample Causal Tuning Algorithm [1]_

    References
    ----------
    .. [1] Biza, Konstantina, Ioannis Tsamardinos, and Sofia Triantafillou.
      "Out-of-Sample Tuning for Causal Discovery."
      IEEE Transactions on Neural Networks and Learning Systems (2022).

    """

    def __init__(self, cv=10, sparsity_penalty=True, thr=0.05, random_state=None):
        """Construct a OutOfSampleCausalTuning.

        Parameters
        ----------
        cv : int, cross-validation generator or an iterable, default=10
            Determines the cross-validation splitting strategy.

        sparsity_penalty : bool
            Whether to enable sparsity penalty.

        thr : float
            Significance threshold of the permutation test of sparsity penalty.

        random_state : None, int or instance of RandomState, default=None
            Controls the permutation test of sparsity penalty when applying
            sparsity penalty.

        Attributes
        ----------
        best_config_index_ : int
            Index of the best configuraion.

        scores_ : dictionary
            Scores of the given configurations.
        """

        self.cv = cv
        self.thr = thr
        self.sparsity_penalty = sparsity_penalty
        self.random_state = random_state

    @property
    def best_config_index_(self):
        check_is_fitted(self)
        return self.best_idx_

    @property
    def scores_(self):
        check_is_fitted(self)
        return self.fit_info_

    def fit(self, data, configs):
        """Search the index of the best configuration given in configs.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The input samples.

        configs : list of dictionary
            A list of configurations specifying the causal discovery
            algorithm and its parameters. Each configuration shall have
            ``model`` unless ``all_others`` is True.
            ``model`` specifies the constructor of a causal discovery algorithm
            such as DirectLiNGAM or ICALiNGAM. ``init_kwargs`` and ``fit_kwargs``
            specify the arguments to be given to the constructor and fit of
            the causal discovery algorithm. Set ``all_others`` to True if you
            want to ignore markov blankets when making predictive models
            for each variable.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """

        # check input args
        data = check_array(data, ensure_min_features=2)
        configs = self._check_configs(configs)

        # check parameters
        cv = check_cv(self.cv)
        thr = check_scalar(
            self.thr,
            "thr",
            float,
            min_val=0,
            max_val=1,
            include_boundaries="neither"
        )
        sp = check_scalar(
            self.sparsity_penalty, "sparsity_penalty", bool
        )
        rs = check_random_state(self.random_state)

        scores = []
        mb_sizes = []
        config_infos = []

        # evaluate each config
        for config in configs:
            config_info = []

            # make model set
            for cv_idx, (train, test) in enumerate(cv.split(data)):
                train_data = data[train]
                test_data = data[test]

                if config["all_others"] is False:
                    graph = self._estimate_graph(train_data, config)
                else:
                    graph = None

                # select explanatories based on the estiamted graph
                exps = self._select_explanatory(graph, data.shape[1])

                # make models of variables
                models = self._make_var_models(train_data, exps, rs)

                # predict
                preds = self._predict_vars(test_data, exps, models)

                cv_info = {
                    "test": test,
                    "graph": graph,
                    "exps": exps,
                    "preds": preds,
                }
                config_info.append(cv_info)

            # evaluate config
            score = self._evaluate_config(data, config_info)

            # calculate average markov blanket size (num of explanatory)
            mb_size = self._get_mb_sizes(config_info)

            scores.append(score)
            mb_sizes.append(mb_size)
            config_infos.append(config_info)

        # select temporal best conf
        best_idx = np.argmax(scores)

        if sp is True:
            best_idx, sp_scores = self._sparsity_penalty(
                data, thr, rs, scores, mb_sizes, config_infos, best_idx
            )
        else:
            sp_scores = None

        self.best_idx_ = best_idx

        fit_info = {}
        fit_info["performance"] = scores
        fit_info["mb_size"] = mb_sizes
        fit_info["sp_score"] = sp_scores
        self.fit_info_ = fit_info

        return self

    def _check_configs(self, configs):
        if not isinstance(configs, list):
            raise TypeError("configs is not a list")

        if len(configs) < 1:
            raise ValueError("configs is empty")

        configs_ = []

        for config_idx, config in enumerate(configs):
            if not isinstance(config, dict):
                raise TypeError(f"configs[{config_idx}] is not a dict")

            # defualt values
            config_ = {
                "all_others": False,
                "model": None,
                "init_kwargs": {},
                "fit_kwargs": {},
            }

            # parse config
            for key, value in config.items():
                if key == "all_others":
                    check_scalar(value, key, bool)
                elif key == "model":
                    if not callable(value):
                        raise TypeError(f"{key} is not callable (config[{config_idx}])")
                elif key == "init_kwargs":
                    if not isinstance(value, dict):
                        raise TypeError(f"{key} is not a dict (config[{config_idx}])")
                elif key == "fit_kwargs":
                    if not isinstance(value, dict):
                        raise TypeError(f"{key} is not a dict (config[{config_idx}])")
                else:
                    raise ValueError(f"{key} is an unknown key (config[{config_idx}])")
                config_[key] = value

            # check
            if config_["all_others"] is False and "model" not in config_.keys():
                raise ValueError(f"model is not specified (config[{config_idx}])")
            elif config_["all_others"] is True:
                config_["model"] = None
                config_["init_kwargs"] = {}
                config_["fit_kwargs"] = {}

            configs_.append(config_)

        return configs_

    def _estimate_graph(self, train_data, config):
        model = config["model"](**config["init_kwargs"])
        model.fit(train_data, **config["fit_kwargs"])

        return model.adjacency_matrix_

    def _get_mb(self, i, graph):
        pa = np.argwhere(~np.isclose(graph[i], 0)).flatten()
        ch = np.argwhere(~np.isclose(graph[:, i], 0)).flatten()

        co_pa = []
        if len(ch) > 0:
            for ch_ in ch:
                co_pa_ = np.argwhere(~np.isclose(graph[ch_], 0)).flatten()
                co_pa.append(co_pa_)
            co_pa = np.concatenate(co_pa)
            co_pa = co_pa[co_pa != i]

        return np.unique([*pa, *ch, *co_pa]).tolist()

    def _select_explanatory(self, graph, n_features):
        exps = []

        for var_idx in range(n_features):
            if graph is not None:
                exp = self._get_mb(var_idx, graph)
            else:
                exp = sorted(set(range(n_features)) - set([var_idx]))
            exps.append(exp)

        return exps

    def _make_var_models(self, train_data, exps, random_state):
        models = []

        for var_idx, exp in enumerate(exps):
            X = train_data[:, exp]
            y = train_data[:, var_idx]

            if len(exp) > 0:
                model = RandomForestRegressor(
                    n_jobs=-1,
                    random_state=random_state,
                )
            else:
                model = DummyRegressor(strategy="mean")

            model.fit(X, y)

            models.append(model)

        return models

    def _predict_vars(self, test_data, exps, models):
        preds = {}

        for var_idx, exp in enumerate(exps):
            X = test_data[:, exp]

            pred = models[var_idx].predict(X)

            preds[var_idx] = pred

        return preds

    def _calc_mutual_info(self, x, x2):
        return -0.5 * np.log(1 - np.corrcoef(x, x2)[0, 1] ** 2)

    def _evaluate_config(self, data, config_info):
        scores = []

        for var_idx in range(data.shape[1]):
            pred = []
            truth = []

            for cv_idx, config_info_ in enumerate(config_info):
                pred.append(config_info_["preds"][var_idx])
                truth.append(data[config_info_["test"], var_idx])

            pred = np.concatenate(pred)
            truth = np.concatenate(truth)

            score = self._calc_mutual_info(truth, pred)

            scores.append(score)

        return np.mean(scores)

    def _get_mb_sizes(self, config_info):
        mb_sizes = []

        for cv_idx, config_info_ in enumerate(config_info):
            for var_idx, exp in enumerate(config_info_["exps"]):
                mb_sizes.append(len(exp))

        return np.mean(mb_sizes)

    def _sparsity_penalty(self, data, thr, rs, scores, mb_sizes, config_infos, best_idx):
        ret = best_idx
        trial_num = 1000

        ps = {}
        for conf_idx, config_info in enumerate(config_infos):
            if conf_idx == best_idx:
                continue

            # permutation test
            t_obs = scores[best_idx] - scores[conf_idx]

            ts = []

            for trial in range(trial_num):
                best_mis = []
                mis = []

                for var_idx in range(data.shape[1]):
                    # concatenate predicted values over CVs
                    best_pred = np.hstack(
                        [cv_info["preds"][var_idx] for cv_info in config_infos[best_idx]]
                    )

                    # concatenate predicted values over CVs
                    pred = np.hstack([cv_info["preds"][var_idx] for cv_info in config_info])

                    # concatenate true values over CVs
                    test_indices = np.hstack([cv_info["test"] for cv_info in config_info])
                    truth = np.hstack(data[test_indices, var_idx])

                    # swap
                    best_pred_pm = rs.permutation(best_pred)
                    pred_pm = rs.permutation(pred)

                    # mutual info with original data
                    best_mi = self._calc_mutual_info(truth, best_pred_pm)
                    mi = self._calc_mutual_info(truth, pred_pm)

                    best_mis.append(best_mi)
                    mis.append(mi)

                t = np.mean(best_mis) - np.mean(mis)
                ts.append(t)

            p = sum(map(lambda t: t > t_obs, ts)) / trial_num
            ps[conf_idx] = p

        # select configuration
        matches = [k for k, v in ps.items() if v > thr]

        if len(matches) > 0:
            matches = [best_idx] + matches

            mbs = [mb_sizes[idx] for idx in matches]
            ret = matches[np.argmin(mbs)]

        return ret, ps
