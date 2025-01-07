#!/usr/bin/env python3


import numpy as np
import scipy.sparse as sps
from cymetrics import RelevanceMetrics
from tqdm import tqdm

import ds_utils
from utils import EPOCHS, MAX_EPOCHS, REL_METRICS_


def eval_toppop(trained_model, data, metrics, Ks, seed, run_path, classifier='gbt') -> dict:
    URM = data[ds_utils.TEST_IN]
    URM_test = data[ds_utils.TEST_OUT]

    n_users, n_items = URM.shape
    batch_size = 50_000

    rel_metrics = RelevanceMetrics([m for m in metrics if m in REL_METRICS_], Ks, URM)

    # Compute top popular items
    URM.data = np.ones_like(URM.data, dtype=np.int8)
    item_ratings = URM.sum(axis=0).A1
    top_items = np.argsort(item_ratings)[-max(Ks):]

    preds = sps.lil_matrix(URM_test.shape, dtype=np.int8)
    preds[:, top_items] = item_ratings[top_items]

    for st_idx in tqdm(range(0, n_users, batch_size), desc=f'Evaluating {run_path}'):
        end_idx = min(st_idx + batch_size, n_users)
        rel_metrics.calculate(URM_test[st_idx: end_idx], preds[st_idx: end_idx].toarray())

    return rel_metrics.values


def train_toppop(seed, path, data, Ks, early_stop_freq=1, early_stop_allow_worse=5, epochs=MAX_EPOCHS,
                 batch_size=500, code_dim=10, lr=1e-1):
    return {EPOCHS: 0}
