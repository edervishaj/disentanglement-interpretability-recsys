#!/usr/bin/env python3


import os

import numpy as np
from cymetrics import RelevanceMetrics
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm

import ds_utils
from utils import (DIS_METRICS_, EPOCHS, MAX_EPOCHS, REL_METRICS_, eval_disen_xai)


def save_puresvd(trained_model, path):
    U, Sigma, VT = trained_model
    np.save(os.path.join(path, 'U'), U)
    np.save(os.path.join(path, 'Sigma'), Sigma)
    np.save(os.path.join(path, 'VT'), VT)


def load_puresvd(path):
    return np.load(os.path.join(path, 'U.npy')), \
        np.load(os.path.join(path, 'Sigma.npy')), \
        np.load(os.path.join(path, 'VT.npy')) 


def eval_puresvd(trained_model, data, metrics, Ks, seed, run_path, classifier='gbt') -> dict:
    URM = data[ds_utils.TEST_IN]
    URM_test = data[ds_utils.TEST_OUT]

    n_users, n_items = URM.shape
    batch_size = 5120

    rel_metrics = RelevanceMetrics([m for m in metrics if m in REL_METRICS_], Ks, URM)

    predictions = []
    train_codes = []

    U, Sigma, VT = trained_model['model']
    user_factors = U
    item_factors = np.dot(np.diag(Sigma), VT)

    for st_idx in tqdm(range(0, n_users, batch_size), desc=f'Evaluating {run_path}'):
        end_idx = min(st_idx + batch_size, n_users)

        _preds = np.dot(user_factors[st_idx: end_idx], item_factors)
        _preds[URM[st_idx: end_idx].nonzero()] = -np.inf

        train_codes.append(user_factors[st_idx: end_idx])

        if URM_test is not None:
            rel_metrics.calculate(URM_test[st_idx: end_idx], _preds)
    
    predictions = np.concatenate(predictions) if len(predictions) > 0 else np.array(predictions)
    train_codes = np.concatenate(train_codes) if len(train_codes) > 0 else np.array(train_codes)

    results = rel_metrics.values
    
    if len(train_codes) > 0 and ds_utils.FACTORS_TRAIN in data and ds_utils.FACTORS_VALID in data:
        dis_metrics = [m for m in metrics if m in DIS_METRICS_]
        valid_codes = []
        valid_URM = data[ds_utils.VALID_TEST_DATA_IN]
        n_users = valid_URM.shape[0]
        for st_idx in tqdm(range(0, n_users, batch_size), desc=f'Evaluating {run_path}'):
            end_idx = min(st_idx + batch_size, n_users)
            valid_codes.append(user_factors[st_idx: end_idx])
        valid_codes = np.concatenate(valid_codes) if len(valid_codes) > 0 else np.array(valid_codes)
        disen_xai_results = eval_disen_xai(dis_metrics, train_codes, data[ds_utils.FACTORS_TRAIN], valid_codes,
                                           data[ds_utils.FACTORS_VALID], seed, run_path, classifier)
        results.update(disen_xai_results)
    return results


def train_puresvd(seed, path, data, metrics, Ks, early_stop_freq=1, early_stop_allow_worse=5, 
                  epochs=MAX_EPOCHS, code_dim=200, batch_size=500, lr=1e-1):
    train_data = data[ds_utils.TRAINING_DATA]

    if 'evaluations' in path:
        # Skip training if a checkpoint exists in evaluation path
        try:
            model_out = load_puresvd(path)
            return {'model': model_out, EPOCHS: 0} 
        except FileNotFoundError as e:
            pass

    model_out = randomized_svd(train_data, n_components=int(code_dim), random_state=seed)
    save_puresvd(model_out, path)
    return {'model': model_out, EPOCHS: 0}

