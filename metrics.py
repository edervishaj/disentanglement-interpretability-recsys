#!/usr/bin/env python3

"""
Largely based on "A Framework for the Quantitative Evaluation of Disentangled Representations"
Paper: https://openreview.net/pdf?id=By-7dz-AZ.

Code adapted from https://github.com/google-research/disentanglement_lib.
"""

import os
import sys
from contextlib import contextmanager

import lime
import numpy as np
import pandas as pd
import shap
import tensorflow._api.v2.compat.v1 as tf1
from joblib import Parallel, delayed, dump, load
from numpy.core.multiarray import array as array
from scipy.spatial.distance import jensenshannon
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

TINY = 1e-12

CLS_SEARCH_SPACE = {
    'logistic': {
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'C': [1.0, 10, 100, 1_000, 10_000]
    },
    'rf': {
        'n_estimators': [10, 20, 50, 100],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2, 4]
    },
    'gbt': {
        'learning_rate': [0.1, 0.01, 0.001],
        'n_estimators': [10, 20, 50, 100],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2, 4]
    }
}

CLS_MODEL_ATTR = {
    'logistic': {
        'model': LogisticRegression(solver='saga'),
        'attr': 'coef_'
    },
    'rf': {
        'model': RandomForestClassifier(),
        'attr': 'feature_importances_'
    },
    'gbt': {
        'model': GradientBoostingClassifier(),
        'attr': 'feature_importances_'
    }
}


def has_cuda():
    return tf1.test.is_gpu_available(cuda_only=True)


@contextmanager
def suppress_stderr():
    # Save the current stderr file descriptor
    original_stderr = sys.stderr
    # Redirect stderr to null
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        # Close the temporary stderr and restore original stderr
        sys.stderr.close()
        sys.stderr = original_stderr


def _entropy(probabilities):
    log_probabilities = np.log(probabilities + TINY) / np.log(probabilities.shape[1] + TINY)
    return -np.multiply(probabilities, log_probabilities).sum(axis=1)


def tune_classifier(cls_type, x_valid, y_valid, run_path, seed) -> list:
    """
    y_train has shape [num_samples x num_factors]
    """
    
    if cls_type == 'logistic':
        x_valid = StandardScaler().fit_transform(x_valid)

    valid_num_factors = y_valid.shape[1]
    best_models = []
    
    # First perform GridSearchCV to find the best models
    best_models_path = os.path.join(run_path, 'grid_search_best_models')
    if not os.path.isdir(best_models_path):
        os.makedirs(best_models_path)
    else:
        if not 'param-search' in run_path:
            for i in range(valid_num_factors):
                model_path = os.path.join(best_models_path, f'{cls_type}_factor_{i}.joblib')
                if os.path.isfile(model_path):
                    best_models.append(load(model_path))
            if len(best_models) == valid_num_factors:
                return best_models

    tqdm_iterator = tqdm(range(valid_num_factors), desc='GridSearchCV', leave=False)
    for i in tqdm_iterator:
        model = clone(CLS_MODEL_ATTR[cls_type]['model']).set_params(random_state=np.random.RandomState(seed))
        search = GridSearchCV(estimator=model, param_grid=CLS_SEARCH_SPACE[cls_type], n_jobs=-1, refit=False, cv=2)
        search.fit(x_valid, y_valid[:, i])
        curr_best_model = clone(CLS_MODEL_ATTR[cls_type]['model']).set_params(
            random_state=np.random.RandomState(seed), **search.best_params_)
        best_models.append(curr_best_model)
        dump(curr_best_model, os.path.join(best_models_path, f'{cls_type}_factor_{i}.joblib'))
    tqdm_iterator.close()
    return best_models


def _compute_importance(cls_type, x_train, y_train, x_valid, y_valid, run_path, seed) -> np.array:
    def _fit(model, attr, x_train, y_train):
        model.fit(x_train, y_train)
        return np.abs(getattr(model, attr)).reshape(-1, 1)
    
    num_factors = y_train.shape[1]
    best_models = tune_classifier(cls_type, x_valid, y_valid, run_path, seed)
    importance_matrix = [
        _fit(best_models[i], CLS_MODEL_ATTR[cls_type]['attr'], x_train, y_train[:, i]) for i in tqdm(range(num_factors))
    ]
    importance_matrix = np.hstack(importance_matrix)
    return importance_matrix    # [num_codes x num_factors]


class Metric:
    def __init__(self, codes: np.array, ground_truth: np.array, valid_codes: np.array, valid_ground_truth: np.array,
                 seed: int, run_path: str, classifier_type='logistic'):
        """
        codes has shape [num_samples x num_codes]
        ground_truth has shape [num_samples x num_factors]
        """
        assert classifier_type in ('logistic', 'svc', 'sgd', 'rf', 'gbt'), \
        f'classifier_type must be a value between (logistic, svc, sgd, rf, gbt), got {classifier_type}'

        self.codes = codes
        self.ground_truth = ground_truth
        self.valid_codes = valid_codes
        self.valid_ground_truth = valid_ground_truth
        self.seed = seed
        self.run_path = run_path
        self.cls_type = classifier_type

        self.codes = StandardScaler().fit_transform(self.codes)
        self.valid_codes = StandardScaler().fit_transform(self.valid_codes)

    def calculate(self) -> float:
        raise NotImplementedError


class DCI(Metric):
    def __init__(self, codes: np.array, ground_truth: np.array, valid_codes: np.array, valid_ground_truth: np.array,
                 seed: int, run_path: str, classifier_type='logistic'):
        super().__init__(codes, ground_truth, valid_codes, valid_ground_truth, seed, run_path, classifier_type)
        self.importance_matrix = None

    def _disentanglement(self) -> float:
        if self.importance_matrix is None:
            if np.any(np.isnan(self.codes)): return 0
            self.importance_matrix = _compute_importance(self.cls_type, self.codes, self.ground_truth, self.valid_codes,
                                                         self.valid_ground_truth, self.run_path, self.seed)
            np.save(os.path.join(self.run_path, 'importance_matrix.npy'), self.importance_matrix)
            
        # convert importance_matrix into probabilities
        norm_importance_matrix = self.importance_matrix / (self.importance_matrix.sum(axis=1).reshape(-1, 1) + TINY)

        code_disentanglement = 1.0 - _entropy(norm_importance_matrix)
        code_importance = self.importance_matrix.sum(axis=1) / self.importance_matrix.sum()
        return np.sum(code_importance * code_disentanglement)

    def _completeness(self) -> float:
        if self.importance_matrix is None:
            if np.any(np.isnan(self.codes)): return 0
            self.importance_matrix = _compute_importance(self.cls_type, self.codes, self.ground_truth, self.valid_codes,
                                                         self.valid_ground_truth, self.run_path, self.seed)
            np.save(os.path.join(self.run_path, 'importance_matrix.npy'), self.importance_matrix)

        # convert importance_matrix into probabilities
        norm_importance_matrix = self.importance_matrix / (self.importance_matrix.sum(axis=0).reshape(1, -1) + TINY)
        
        factor_completeness = 1.0 - _entropy(norm_importance_matrix.T)
        factor_importance = self.importance_matrix.sum(axis=0) / self.importance_matrix.sum()
        return np.sum(factor_importance * factor_completeness)

    def calculate(self, measure='disentanglement') -> float:
        return self._completeness() if measure == 'completeness' else self._disentanglement()


class SHAP(Metric):
    def __init__(self, codes: np.array, ground_truth: np.array, valid_codes: np.array, valid_ground_truth: np.array,
                 seed: int, run_path: str, classifier_type='logistic'):
        super().__init__(codes, ground_truth, valid_codes, valid_ground_truth, seed, run_path, classifier_type)

    def calculate(self) -> float:
        best_models = tune_classifier(self.cls_type, self.valid_codes, self.valid_ground_truth, self.run_path,
                                      self.seed)

        num_factors = self.ground_truth.shape[1]

        # Fit best models to each factor
        importance_matrix = [self._fit(best_models[i], i) for i in tqdm(range(num_factors))]
        importance_matrix = np.hstack(importance_matrix)
        np.save(os.path.join(self.run_path, 'shap_importance_matrix.npy'), importance_matrix)

        # Convert columns to probabilities
        importance_matrix = importance_matrix / (importance_matrix.sum(axis=0) + TINY)

        # Compute the column-wise Jensen-Shannon divergence
        divergences = np.full((importance_matrix.shape[1], importance_matrix.shape[1]), -0.01)
        for i in range(importance_matrix.shape[1]):
            for j in range(i+1, importance_matrix.shape[1]):
                divergences[i, j] = jensenshannon(importance_matrix[:, i], importance_matrix[:, j], base=2)
        
        return np.mean(divergences[divergences >= 0.0])

    def _fit(self, model, index) -> np.array:
        ground_truth = self.ground_truth[:, index]
        model.fit(self.codes, ground_truth)

        if self.cls_type in ('rf', 'gbt'):
            cls = shap.GPUTreeExplainer if has_cuda() else shap.TreeExplainer
            explainer = cls(model, self.codes)
        else:
            explainer = shap.LinearExplainer(model, self.codes)
        
        # Need to suppress stderr because SHAP internal C++ code writes a progressbar to stderr
        with suppress_stderr():
            shap_values = explainer.shap_values(self.codes)

        return pd.DataFrame(shap_values).abs().mean().values.reshape(-1, 1)


class LIME(Metric):
    def __init__(self, codes: np.array, ground_truth: np.array, valid_codes: np.array, valid_ground_truth: np.array, 
                 seed: int, run_path: str, classifier_type='logistic'):
        super().__init__(codes, ground_truth, valid_codes, valid_ground_truth, seed, run_path, classifier_type)

    def calculate(self) -> float:
        best_models = tune_classifier(self.cls_type, self.valid_codes, self.valid_ground_truth, self.run_path,
                                      self.seed)

        num_factors = self.ground_truth.shape[1]

        # Fit best models to each factor
        importance_matrix = Parallel(n_jobs=-1)(delayed(self._fit)(best_models[i], i) for i in range(num_factors))
        importance_matrix = np.hstack(importance_matrix)
        np.save(os.path.join(self.run_path, 'lime_importance_matrix.npy'), importance_matrix)

        # Convert columns to probabilities
        importance_matrix = importance_matrix / (importance_matrix.sum(axis=0) + TINY)

        # Compute the column-wise Jensen-Shannon divergence
        divergences = np.full((importance_matrix.shape[1], importance_matrix.shape[1]), -0.01)
        for i in range(importance_matrix.shape[1]):
            for j in range(i+1, importance_matrix.shape[1]):
                divergences[i, j] = jensenshannon(importance_matrix[:, i], importance_matrix[:, j], base=2)
        
        return np.mean(divergences[divergences >= 0.0])
    
    def _fit(self, model, index) -> np.array:
        ground_truth = self.ground_truth[:, index]
        model.fit(self.codes, ground_truth)

        # Necessary to speed up LIME, 5_000 is the default value.
        num_samples = int(0.1 * len(self.codes)) if len(self.codes) < 50_000 else 5_000

        explainer = lime.lime_tabular.LimeTabularExplainer(self.codes)
        lime_values = []
        for code in self.codes:
            explanations = explainer.explain_instance(data_row=code, predict_fn=model.predict_proba, num_features=self.codes.shape[1], num_samples=num_samples).as_map()[1]
            explanations = [weight for _, weight in sorted(explanations, key=lambda tup: tup[0])]
            lime_values.append(explanations)
        return pd.DataFrame(lime_values).abs().mean().values.reshape(-1, 1)
