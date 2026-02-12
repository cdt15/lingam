"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class LEWIS(object):
    """LEWIS explainer for computing necessity, sufficiency, and necessity-and-sufficiency scores. [1]_ [2]_

    References
    ----------
    .. [1] Sainyam Galhotra, Romila Pradhan, Babak Salimi (2021).
       Explaining Black-Box Algorithms Using Probabilistic Contrastive Counterfactuals.
       SIGMOD '21: International Conference on Management of Data, Virtual Event, China, June 20-25, 2021.
    .. [2] https://sainyamgalhotra.github.io/lewis.zip
    """

    def __init__(self, epsilon=1e-10, random_state=0):
        """Initialize LEWIS explainer.

        Parameters
        ----------
        epsilon : float, optional (default=1e-10)
            Small constant to avoid division by zero.
        random_state : int, optional (default=0)
            Random state for reproducibility.
        """
        self._model_map = {}
        self._epsilon = epsilon
        self._random_state = random_state

    def _get_prob(
        self, df, conditional_names, conditional_values, target_names, target_values
    ):
        """
        Estimate conditional probability.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe.
        conditional_names : list of str
            List of feature names for conditioning.
        conditional_values : list
            List of values corresponding to the conditional features.
        target_names : list of str
            List of target column names (typically one element).
        target_values : list
            List of values that the target should match.

        Returns
        -------
        prob : float
            Estimated conditional probability.
        """
        # Check model cache
        cache_key = (
            tuple(conditional_names),
            tuple(target_names),
            tuple(target_values),
        )
        if cache_key in self._model_map:
            model = self._model_map[cache_key]
            prediction = model.predict_proba([conditional_values])[0][1]
            return float(prediction)

        # Label target values
        mask = np.ones(len(df), dtype=bool)
        for t, v in zip(target_names, target_values):
            mask &= df[t].astype(int) == int(v)
        new_lst = mask.astype(int)
        count = new_lst.sum()

        # If no conditions, simply return the occurrence rate of target values
        if len(conditional_names) == 0:
            return count * 1.0 / df.shape[0]

        # If all targets have the same value, probability is deterministic
        if len(list(set(new_lst))) == 1:
            if new_lst[0] == 1:
                return 1
            else:
                return 0

        X = df[conditional_names].values

        model = RandomForestClassifier(random_state=self._random_state)
        model.fit(X, new_lst)
        self._model_map[cache_key] = model
        prediction = model.predict_proba([conditional_values])[0][1]
        return float(prediction)

    def get_scores(
        self,
        df,
        x_names,
        x_values,
        x_prime_values,
        o_name,
        k_names=[],
        k_values=[],
        c_names=[],
    ):
        """Compute LEWIS explanation scores (Necessity, Sufficiency, and Necessity-and-Sufficiency).

        Parameters
        ----------
        df : pandas.DataFrame
            Input data frame.
        x_names : list of str
            Name of the attribute (or set of attributes) X under causal evaluation.
        x_values : list
            Target (intervened) value x of X, typically representing an improved or alternative value.
        x_prime_values : list
            Baseline or contrastive value x' of X against which x is compared.
        o_name : str
            Name of the outcome variable produced by the black-box model.
        k_names : list of str, optional (default=[])
            Names of contextual variables defining the conditioning set K (used for global, local, or contextual explanations).
        k_values : list, optional (default=[])
            Values corresponding to k_names, forming the concrete context k.
        c_names : list of str, optional (default=[])
            Names of adjustment variables C satisfying the backdoor criterion.

        Returns
        -------
        necessity : float
            The necessity score.
        sufficiency : float
            The sufficiency score.
        necessity_and_sufficiency : float
            The necessity and sufficiency score.
        """
        # Check parameters
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        if not isinstance(x_names, list) or not all(isinstance(v, str) for v in x_names):
            raise ValueError("x_names must be a list of strings")
        if not isinstance(x_values, list):
            raise ValueError("x_values must be a list")
        if not isinstance(x_prime_values, list):
            raise ValueError("x_prime_values must be a list")
        if not isinstance(o_name, str):
            raise ValueError("o_name must be a string")
        if not isinstance(k_names, list) or not all(isinstance(v, str) for v in k_names):
            raise ValueError("k_names must be a list of strings")
        if not isinstance(k_values, list):
            raise ValueError("k_values must be a list")
        if not isinstance(c_names, list) or not all(isinstance(v, str) for v in c_names):
            raise ValueError("c_names must be a list of strings")
        if len(x_names) != len(x_values):
            raise ValueError(
                f"x_names ({len(x_names)}) and x_values ({len(x_values)}) must have the same length"
            )
        if len(x_names) != len(x_prime_values):
            raise ValueError(
                f"x_names ({len(x_names)}) and x_prime_values ({len(x_prime_values)}) must have the same length"
            )
        if len(k_names) != len(k_values):
            raise ValueError(
                f"k_names ({len(k_names)}) and k_values ({len(k_values)}) must have the same length"
            )

        c_names_ = [v for v in c_names if v not in k_names]

        if len(c_names_) == 0:
            c_values = [tuple()]
        elif len(c_names_) == 1:
            c_values = [(val,) for val in df[c_names_[0]].unique()]
        else:
            c_values = list(df.groupby(c_names_).groups.keys())

        p_o_doxk = 0
        p_o_doxpk = 0
        nesuf = 0
        self._model_map = {}
        for c_value in c_values:
            cxk_names = k_names + c_names_ + x_names
            cxk_values = k_values + list(c_value) + x_values
            cxpk_values = k_values + list(c_value) + x_prime_values

            # P[o|cxk]
            p_o_cxk = self._get_prob(df, cxk_names, cxk_values, [o_name], [1])

            # P[o|cx'k]
            if p_o_cxk > self._epsilon:
                p_o_cxpk = self._get_prob(df, cxk_names, cxpk_values, [o_name], [1])
            else:
                p_o_cxpk = 0
                continue

            if len(c_names_) > 0:
                # P[c,k]
                p_ck = self._get_prob(df, k_names, k_values, c_names_, c_value)

                # P[c|xk]
                xk_names = k_names + x_names
                xk_values = k_values + x_values
                p_c_xk = self._get_prob(df, xk_names, xk_values, c_names_, c_value)

                # P[c|x'k]
                xpk_values = k_values + x_prime_values
                p_c_xpk = self._get_prob(df, xk_names, xpk_values, c_names_, c_value)
            else:
                p_ck = 1
                p_c_xpk = 1
                p_c_xk = 1

            # P[o'|cx'k]
            p_op_cxpk = 1 - p_o_cxpk

            p_o_doxk += p_o_cxk * p_c_xpk  # P[o|do(x),k]
            p_o_doxpk += p_op_cxpk * p_c_xk  # P[o|do(x'),k]
            nesuf += (p_o_cxk - p_o_cxpk) * p_ck  # P[o|do(x),k]-P[o|do(x'),k]

        # P[o|k]
        if len(k_names) > 0:
            p_o_k = self._get_prob(df, k_names, k_values, [o_name], [1])
        else:
            p_o_k = df[df[o_name] == 1].shape[0] * 1.0 / df.shape[0]

        xk_names = k_names + x_names
        xk_values = k_values + x_values
        xpk_values = k_values + x_prime_values

        # P[o|xk], P[o'|xk]
        p_o_xk = self._get_prob(df, xk_names, xk_values, [o_name], [1])
        p_op_xk = 1 - p_o_xk

        # P[o|x'k], P[o'|x'k]
        p_o_xpk = self._get_prob(df, xk_names, xpk_values, [o_name], [1])
        p_op_xpk = 1 - p_o_xpk

        if p_o_xk > self._epsilon:
            nec = (p_o_doxpk - p_op_xk) * 1.0 / p_o_xk
        else:
            nec = 0.0

        if p_op_xpk > self._epsilon:
            suf = (p_o_doxk - p_o_xpk) * 1.0 / p_op_xpk
        else:
            suf = 0.0

        return (
            float(np.clip(nec, 0.0, 1.0)),
            float(np.clip(suf, 0.0, 1.0)),
            float(np.clip(nesuf, 0.0, 1.0)),
        )
