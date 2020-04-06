import os

import numpy as np
import pandas as pd

from lingam.base import _BaseLiNGAM
from lingam.causal_effect import CausalEffect

class DummyLiNGAM(_BaseLiNGAM):
    def __init__(self, random_state=None, causal_order=None, adjacency_matrix=None):
        super().__init__(random_state)
        self._causal_order = causal_order
        self._adjacency_matrix = adjacency_matrix
    def fit(self, X):
        pass

class DummyPrediction(object):
    def __init__(self, reval_predict=None, coef=None, intercept=None):
        self._reval_predict = reval_predict
        self._coef = coef
        self._intercept = intercept
    def predict(self, X):
        return self._reval_predict
    @property
    def coef_(self):
        return self._coef
    @property
    def intercept_(self):
        return self._intercept

def test_estimate_effects_on_prediction_success():
    # causal direction: x0 --> x1, x0 --> x2, x1 --> x2
    x0 = np.random.uniform(size=1000)
    x1 = -1.0*x0 + np.random.uniform(size=1000)
    x2 = 3.0*x0 + 2.0*x1 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2]).T, columns=['x0', 'x1', 'x2'])

    # specified LiNGAM for causal model
    model = DummyLiNGAM(causal_order=[0, 1, 2],
        adjacency_matrix=np.array([
            [ 0., 0., 0.],
            [-1., 0., 0.],
            [ 3., 2., 0.],
        ]))
    ce = CausalEffect(model)
    pred_model = DummyPrediction(reval_predict=np.array([0]))
    effects = ce.estimate_effects_on_prediction(X, 2, pred_model)
    assert effects.shape == (3, 2)

    En = np.array([0.1, 0.1, 0.1])
    effects = ce._get_propagated_effects(En, 0, 1)
    assert effects[0] == 1
    assert effects[1] == -0.9
    assert effects[2] == (3.0+(-1.8)+0.1)

    # specified np.array for causal model
    ce = CausalEffect(np.array([[0., -1., 0.], [ 0., 0., 0.], [ 3., 2., 0.]]))
    pred_model = DummyPrediction(reval_predict=np.array([0]))
    effects = ce.estimate_effects_on_prediction(X, 2, pred_model)
    assert effects.shape == (3, 2)

    # specified list for causal model
    ce = CausalEffect([[0., -1., 0.], [ 0., 0., 0.], [ 3., 2., 0.]])
    pred_model = DummyPrediction(reval_predict=np.array([0]))
    effects = ce.estimate_effects_on_prediction(X, 2, pred_model)
    assert effects.shape == (3, 2)

def test_estimate_effects_on_prediction_invalid_input():
    # causal direction: x0 --> x1, x0 --> x2, x1 --> x2
    x0 = np.random.uniform(size=1000)
    x1 = -1.0*x0 + np.random.uniform(size=1000)
    x2 = 3.0*x0 + 2.0*x1 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2]).T, columns=['x0', 'x1', 'x2'])

    # Not causal model
    try:
        ce = CausalEffect(1)
        pred_model = DummyPrediction(reval_predict=np.array([0]))
        ce.estimate_effects_on_prediction(X, 2, pred_model)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Not matrix
    try:
        ce = CausalEffect(np.array([0, 1, 2]))
        pred_model = DummyPrediction(reval_predict=np.array([0]))
        ce.estimate_effects_on_prediction(X, 2, pred_model)
    except ValueError:
        pass
    else:
        raise AssertionError

    try:
        ce = CausalEffect([0, 1, 2])
        pred_model = DummyPrediction(reval_predict=np.array([0]))
        ce.estimate_effects_on_prediction(X, 2, pred_model)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Not square matrix
    try:
        ce = CausalEffect(np.array([[0, 1, 2], [0, 0, 0]]))
        pred_model = DummyPrediction(reval_predict=np.array([0]))
        ce.estimate_effects_on_prediction(X, 2, pred_model)
    except ValueError:
        pass
    else:
        raise AssertionError

    try:
        ce = CausalEffect([[0, 1, 2], [0, 0, 0]])
        pred_model = DummyPrediction(reval_predict=np.array([0]))
        ce.estimate_effects_on_prediction(X, 2, pred_model)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Not triangular matrix
    try:
        ce = CausalEffect(np.array([[0, 1, 2], [1, 1, 1], [2, 3, 4]]))
        pred_model = DummyPrediction(reval_predict=np.array([0]))
        ce.estimate_effects_on_prediction(X, 2, pred_model)
    except ValueError:
        pass
    else:
        raise AssertionError

    try:
        ce = CausalEffect([[0, 1, 2], [1, 1, 1], [2, 3, 4]])
        pred_model = DummyPrediction(reval_predict=np.array([0]))
        ce.estimate_effects_on_prediction(X, 2, pred_model)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Not prediction model
    try:
        ce = CausalEffect(np.array([[0, 0, 0], [1, 0, 0], [2, 2, 0]]))
        pred_model = [0]
        ce.estimate_effects_on_prediction(X, 2, pred_model)
    except ValueError:
        pass
    else:
        raise AssertionError


def test_estimate_optimal_intervention_success():
    # causal direction: x0 --> x1, x0 --> x2, x1 --> x2
    x0 = np.random.uniform(size=1000)
    x1 = -1.0*x0 + np.random.uniform(size=1000)
    x2 = 3.0*x0 + 2.0*x1 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2]).T, columns=['x0', 'x1', 'x2'])

    # specified LiNGAM for causal model
    model = DummyLiNGAM(causal_order=[0, 1, 2],
        adjacency_matrix=np.array([
            [ 0., 0., 0.],
            [-1., 0., 0.],
            [ 3., 2., 0.],
        ]))
    ce = CausalEffect(model)
    pred_model = DummyPrediction(reval_predict=np.array([1]), coef=np.array([1, 2]), intercept=0)
    optimal_intervention = ce.estimate_optimal_intervention(X, 2, pred_model, 0, 0)
    assert optimal_intervention > 0.95 and optimal_intervention < 1.05

    # specified np.array for causal model
    ce = CausalEffect(np.array([[ 0., 0., 0.], [-1., 0., 0.], [ 3., 2., 0.]]))
    pred_model = DummyPrediction(reval_predict=np.array([1]), coef=np.array([1, 2]), intercept=0)
    optimal_intervention = ce.estimate_optimal_intervention(X, 2, pred_model, 0, 0)
    assert optimal_intervention > 0.95 and optimal_intervention < 1.05

    # specified list for causal model
    ce = CausalEffect([[ 0., 0., 0.], [-1., 0., 0.], [ 3., 2., 0.]])
    pred_model = DummyPrediction(reval_predict=np.array([1]), coef=np.array([1, 2]), intercept=0)
    optimal_intervention = ce.estimate_optimal_intervention(X, 2, pred_model, 0, 0)
    assert optimal_intervention > 0.95 and optimal_intervention < 1.05

def test_estimate_optimal_intervention_invalid_input():
    # causal direction: x0 --> x1, x0 --> x2, x1 --> x2
    x0 = np.random.uniform(size=1000)
    x1 = -1.0*x0 + np.random.uniform(size=1000)
    x2 = 3.0*x0 + 2.0*x1 + np.random.uniform(size=1000)
    X = pd.DataFrame(np.array([x0, x1, x2]).T, columns=['x0', 'x1', 'x2'])

    # Not causal model
    try:
        ce = CausalEffect(1)
        pred_model = DummyPrediction(reval_predict=np.array([0]))
        ce.estimate_optimal_intervention(X, 2, pred_model, 0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Not matrix
    try:
        ce = CausalEffect(np.array([0, 1, 2]))
        pred_model = DummyPrediction(reval_predict=np.array([0]))
        ce.estimate_optimal_intervention(X, 2, pred_model, 0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError

    try:
        ce = CausalEffect([0, 1, 2])
        pred_model = DummyPrediction(reval_predict=np.array([0]))
        ce.estimate_optimal_intervention(X, 2, pred_model, 0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Not square matrix
    try:
        ce = CausalEffect(np.array([[0, 1, 2], [0, 0, 0]]))
        pred_model = DummyPrediction(reval_predict=np.array([0]))
        ce.estimate_optimal_intervention(X, 2, pred_model, 0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError

    try:
        ce = CausalEffect([[0, 1, 2], [0, 0, 0]])
        pred_model = DummyPrediction(reval_predict=np.array([0]))
        ce.estimate_optimal_intervention(X, 2, pred_model, 0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Not triangular matrix
    try:
        ce = CausalEffect(np.array([[0, 1, 2], [1, 1, 1], [2, 3, 4]]))
        pred_model = DummyPrediction(reval_predict=np.array([0]))
        ce.estimate_optimal_intervention(X, 2, pred_model, 0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError

    try:
        ce = CausalEffect([[0, 1, 2], [1, 1, 1], [2, 3, 4]])
        pred_model = DummyPrediction(reval_predict=np.array([0]))
        ce.estimate_optimal_intervention(X, 2, pred_model, 0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError

    # Not prediction model
    try:
        ce = CausalEffect(np.array([[0, 0, 0], [1, 0, 0], [2, 2, 0]]))
        pred_model = [0]
        ce.estimate_optimal_intervention(X, 2, pred_model, 0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError
