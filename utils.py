#!/usr/bin/env python3


import os
import shutil

import numpy as np
import scipy.sparse as sps
import tensorflow._api.v2.compat.v1 as tf1
from tqdm import tqdm

from cymetrics import RelevanceMetrics
from metrics import DCI, LIME, SHAP

MAX_EPOCHS = 500
EPOCHS = 'epochs'
CLS = 'class'
FACTORS_FN = 'factors_fn'
TRAIN_FN = 'train_fn'
EVAL_FN = 'eval_fn'
DATA_FN = 'data_fn'
PARAM_SPACE = 'space'
CODE_DIM = 'code_dim'
METRICS = 'metrics'
KS = 'K'

REL_METRICS_ = ('ndcg', 'recall', 'mrr', 'coverage')

DIS_METRICS_ = {
    'disentanglement': DCI,
    'completeness': DCI,
    'shap': SHAP,
    'lime': LIME
}


def clean_path(path):
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry.path)
            else:
                os.remove(entry.path)


def tf1_save(tf_saver: tf1.train.Saver, session: tf1.Session, path: str) -> None:
    tf_saver.save(session, '{}/chkpt'.format(path), write_meta_graph=False, write_state=False)


def tf1_load(tf_saver: tf1.train.Saver, session: tf1.Session, path: str) -> None:
    tf_saver.restore(session, '{}/chkpt'.format(path))


def score_vae(run_path, model_name, trained_model, URM, URM_test=None, metrics=None, Ks=None, compute_output=False):
    sess = trained_model['sess']
    output_op = trained_model['output_op']
    code_op = trained_model['code_op']
    model = trained_model[model_name]

    n_users, n_items = URM.shape
    batch_size = 5120

    predictions = []
    codes = []

    rel_metrics = [m for m in metrics if m in REL_METRICS_]
    ops = [output_op, code_op] if len(rel_metrics) > 0 else [code_op]
    rel_metrics = RelevanceMetrics(rel_metrics, Ks, URM)

    for st_idx in tqdm(range(0, n_users, batch_size), desc=f'Evaluating {run_path}'):
        end_idx = min(st_idx + batch_size, n_users)
        x = URM[st_idx: end_idx]
        if sps.isspmatrix(x):
            x = x.toarray()
        x = x.astype('float32')

        res = sess.run(ops, feed_dict={model.input_ph: x})
        codes.append(res[-1])

        if len(ops) > 1:
            _preds = res[0]
            _preds[x != 0] = -np.inf

            if URM_test is not None:
                rel_metrics.calculate(URM_test[st_idx: end_idx], _preds)

            if compute_output:
                predictions.append(_preds)

    predictions = np.concatenate(predictions) if len(predictions) > 0 else np.array(predictions)
    codes = np.concatenate(codes) if len(codes) > 0 else np.array(codes)
    return rel_metrics.values, predictions, codes


def eval_disen_xai(metrics: list, codes: np.array, ground_truth: np.array, valid_codes: np.array, valid_gt: np.array,
                   seed: int, run_path: str, classifier_type='gbt', verbose=True) -> dict:
    results = {}
    metric_obj = {}
    for m in metrics:
        cls = DIS_METRICS_[m]
        if cls not in metric_obj:
            metric_obj[cls] = cls(codes, ground_truth, valid_codes, valid_gt, seed, run_path, classifier_type)

        if verbose:
            print(f'Calculating {metric_obj[cls].__class__.__name__}...', end='')

        if cls == DCI:
            results[m] = metric_obj[cls].calculate(measure=m)
        else:
            results[m] = metric_obj[cls].calculate()

        if verbose:
            print('Done')
    return results
