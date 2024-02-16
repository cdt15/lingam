import numpy as np
import pandas as pd

from sklearn.utils import check_array
from sklearn.base import is_regressor, is_classifier

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def visualize_nonlinear_causal_effect(
    X,
    cd_result,
    estimator,
    cause_name,
    effect_name,
    cause_positions=None,
    percentile=None,
    fig=None,
    boxplot=False,
):
    """Visualize non-linear causal effect.

    Parameters
    ----------
    X : pandas.DataFrame, shape (n_samples, n_features)
        Training data used to obtain cd_result.
    cd_result : array-like with shape (n_features, n_features) or BootstrapResult
        Adjacency matrix or BootstrapResult. These are the results of a causal discovery.
    estimator : estimator object
        ``estimator`` used for non-linear regression.
        Regression with ``estimator`` using ``cause_name`` and covariates as explanatory
        variables and ``effect_name`` as objective variable.
        Those covariates are searched for in ``cd_result``.
    cause_name : str
        The name of the cause variable.
    effect_name : str
        The name of the effect variable.
    cause_positions : array-like, optional (default=None)
        List of positions from which causal effects are calculated.
        By default, ``cause_positions`` stores the position at which the value range of X is divided
        into 10 equal parts.
    percentile : array-like, optional (default=None)
        A tuple consisting of three percentile values. Each value must be greater
        than 0 and less than 100. By default, (95, 50, 5) is set.
    fig : plt.Figure, optional (default=None)
        If ``fig`` is given, draw a figure in ``fig``. If not given, plt.fig
        is prepared internally.
    boxplot : boolean, optional (default=False)
        If True, draw a box plot instead of a scatter plot for each ``cause_positions``.

    Return
    ------
    fig : plt.Figure
        Plotted figure.
    """

    # check arguments
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be pandas.DataFrame.")

    if not _is_bootstrap_result(cd_result) and not _is_adjacency_matrix(
        cd_result, X.shape[1]
    ):
        raise TypeError(
            "cd_result must be lingam.bootstrap.BootstrapResult or array-like."
        )

    if cause_name not in X.columns:
        raise ValueError("cause_name is not exsit in X.columns.")

    if X[cause_name].dtype == "category":
        raise ValueError("cause_name must not be categorical.")

    if effect_name not in X.columns:
        raise ValueError("effect_name is not exsit in X.columns.")

    is_discrete = X[effect_name].dtype == "category"
    _check_estimator(estimator, is_discrete)

    if cause_positions is not None:
        cause_positions = check_array(
            cause_positions, dtype=None, ensure_2d=False
        ).flatten()
        if None in cause_positions:
            raise TypeError("cause_positions must not include None.")

    if percentile is not None:
        if not isinstance(percentile, list) or len(percentile) != 3:
            raise TypeError("percentile must be a list of 3 floats.")
        if not all(map(lambda x: 0 <= x and x <= 100, percentile)):
            raise TypeError(
                "Elements of percentile must be between 0 and 100 inclusive."
            )

    if fig is not None:
        if not isinstance(fig, Figure):
            raise TypeError("fig must be matplotlib.figure.Figure.")

    # set initial values
    if cause_positions is None:
        cause_positions = np.linspace(X[cause_name].min(), X[cause_name].max(), 10)

    if fig is None:
        fig = plt.figure()

    cause_index = list(X.columns).index(cause_name)
    effect_index = list(X.columns).index(effect_name)

    effect_classes = [None] if not is_discrete else np.unique(X[effect_name])

    if percentile is None:
        percentile = (95, 50, 5)

    # calculate causal effects
    if _is_bootstrap_result(cd_result):
        adj_matrices = cd_result.adjacency_matrices_
        index_list = cd_result.resampled_indices_
        if index_list is None:
            index_list = [X.index]
        disable_percentile = False
    else:
        adj_matrices = [cd_result]
        index_list = [X.index]
        disable_percentile = True

    effect_info = _calculate(
        effect_classes,
        adj_matrices,
        index_list,
        cause_index,
        effect_index,
        estimator,
        cause_positions,
        X,
        is_discrete,
    )

    # visualize
    fig = _draw(
        effect_info,
        cause_positions,
        percentile,
        fig,
        cause_name,
        effect_name,
        boxplot,
        disable_percentile,
    )

    return fig


def _is_bootstrap_result(cd_result):
    name = type(cd_result).__module__ + "." + type(cd_result).__name__
    if name != "lingam.bootstrap.BootstrapResult":
        return False
    return True


def _is_adjacency_matrix(cd_result, col_num):
    check_array(cd_result)

    if cd_result.shape[0] != col_num or cd_result.shape[1] != col_num:
        return False
    return True


def _check_estimator(estimator, is_discrete):
    if is_discrete is True and is_regressor(estimator):
        raise TypeError("X[effect_name] is discrete but the estimator is a regressor.")
    elif is_discrete is False and is_classifier(estimator):
        raise TypeError(
            "X[effect_name] is not discrete but the estimator is a classifier."
        )

    if is_classifier(estimator):
        try:
            func = getattr(estimator, "predict_proba")
            if not callable(func):
                raise Exception
        except Exception:
            raise RuntimeError("Classification models must have predict_proba().")


def _calculate(
    effect_classes,
    adj_matrices,
    index_list,
    cause_index,
    effect_index,
    estimator,
    cause_positions,
    X,
    is_discrete,
):
    # effect_classes is [None] if X.iloc[effect_index] is continuous
    effect_info = {effect_class: {} for effect_class in effect_classes}
    for c in effect_info.keys():
        effect_info[c] = {p: [] for p in cause_positions}

    for m, sub_index in zip(adj_matrices, index_list):
        # parents of cause except effect
        covariate = np.where(m[cause_index, :] == 1)[0].tolist()
        covariate = np.delete(covariate, np.argwhere(covariate == effect_index))

        # train
        exp = np.sort([cause_index] + covariate.tolist())
        exp = np.delete(exp, np.argwhere(exp == effect_index))
        X_exp = pd.get_dummies(X.iloc[sub_index, exp])
        X_obj = X.iloc[sub_index, effect_index]
        estimator.fit(X_exp, X_obj)

        # calculate causal effects
        for interv_val in cause_positions:
            # intervention
            X_ = X.iloc[sub_index, :].copy()
            X_.iloc[:, cause_index] = interv_val
            X_exp2 = pd.get_dummies(X_.iloc[:, exp])

            if not is_discrete:
                effect_values = estimator.predict(X_exp2)

                v = np.mean(effect_values)
                effect_info[None][interv_val].append(v)
            else:
                effect_values = estimator.predict_proba(X_exp2)

                for i, c in enumerate(estimator.classes_):
                    v = np.mean(effect_values[:, i])
                    effect_info[c][interv_val].append(v)

    return effect_info


def _draw(
    effect_info,
    cause_positions,
    percentile,
    fig,
    cause_name,
    effect_name,
    is_box,
    disable_percentile,
):
    if disable_percentile is False:
        alpha = 0.5
        colors = ["red", "green", "blue"]
    else:
        percentile = [50]
        alpha = 1.0
        colors = ["green"]

    for i in range(len(effect_info)):
        fig.add_subplot(1, len(effect_info), i + 1)

    axes = np.array(fig.get_axes()).flatten()

    for ax, (class_name, effects) in zip(axes, effect_info.items()):
        per_plot = {per: [] for per in percentile}

        for interv_val, values in effects.items():
            if not is_box:
                ax.scatter(
                    interv_val * np.ones(len(values)),
                    values,
                    color="black",
                    alpha=alpha,
                    facecolor="none",
                )
            else:
                ax.boxplot(values, positions=[interv_val], manage_ticks=False)

            per_points = np.percentile(values, percentile)
            for per_point, percentile_ in zip(per_points, percentile):
                per_plot[percentile_].append(per_point)

        ax.set_xlabel(cause_name)
        if class_name is None:
            ax.set_ylabel(f"causal effect {effect_name}")
        else:
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"class {class_name}")
            ax.set_ylabel("probability")

        for (per, values), color in zip(per_plot.items(), colors):
            label = f"{per}%" if disable_percentile is False else None
            ax.plot(cause_positions, values, color=color, label=label)

    return fig
