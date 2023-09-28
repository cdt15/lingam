import os
import sys
import subprocess
import tempfile
import shutil
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_scalar
from sklearn.utils import check_array
from sklearn.utils import check_random_state


def predict_adaptive_logistic_regression(X, predictors, target, gamma=1.0, random_seed=None):
    """Predict with Adaptive logistic regression.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples
        and n_features is the number of features.
    predictors : array-like, shape (n_predictors)
        Indices of predictor variable.
    target : int
        Index of target variable. X[:, target] is treated as
        a discrete variable.
    gamma : float (optional)
        A parameter for penalty.factor.
    random_seed : int (optinal)
        A random seed.

    Returns
    -------
    coef : array-like, shape (classes, n_features)
        Coefficients of predictor variable.
    """
    # check arguments
    X = check_array(X, ensure_min_features=2)

    predictors = check_array(predictors, dtype=int, ensure_2d=False)

    target = check_scalar(target, "target", int)

    if not isinstance(gamma, (int, float)) or gamma <= 0:
        raise ValueError("gamma is invalid.")
    
    if random_seed is not None:
        random_seed = check_scalar(random_seed, "random_seed", int)

    # check values
    diff = set(predictors) - set(range(X.shape[1]))
    if len(diff) > 0:
        raise ValueError("predictors is invalid.")

    if target not in range(X.shape[1]):
        raise ValueError("target is invalid.")

    # data
    X_ = X[:, predictors]
    y = X[:, target]
        
    try:
        temp_dir = tempfile.mkdtemp()

        path = os.path.join(os.path.dirname(__file__), "adaptive_logistic_regression.r")

        args = [f"--temp_dir={temp_dir}"]
        args += [f"--gamma={gamma}"]
        if random_seed is not None:
            args += [f"--rs={random_seed}"]

        np.savetxt(os.path.join(temp_dir, "X.csv"), X_, delimiter=",")
        np.savetxt(os.path.join(temp_dir, "y.csv"), y, delimiter=",")

        # run
        ret = subprocess.run(["Rscript", path, *args], capture_output=True)
        if ret.returncode != 0:
            print(ret.stdout.decode())
            print(ret.stderr.decode())

        # retrieve result
        coefs = pd.read_csv(os.path.join(temp_dir, "coefs.csv"))
    except FileNotFoundError as e:
        raise RuntimeError("Rscript is not found.")
    except BaseException as e:
        raise RuntimeError(str(e))
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    if ret.returncode == 2:
        raise RuntimeError("glmnet is not installed.")
    elif ret.returncode != 0:
        raise RuntimeError("retcode={}\nstdout={}\nstderr={}".format(ret.returncode, ret.stdout.decode(), ret.stderr.decode()))
    
    coefs.index = coefs["Unnamed: 0"].values
    coefs = coefs.drop(["Unnamed: 0"], axis=1)
    coefs = coefs.drop(["(Intercept)"])
    coefs = coefs.T.values

    return coefs

