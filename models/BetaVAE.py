#!/usr/bin/env python3

from functools import partial

import numpy as np
import scipy.sparse as sps
import tensorflow._api.v2.compat.v1 as tf1
from tqdm import tqdm

import ds_utils
from models.helpers import EarlyStopper
from models.MultiVAE import MultiVAE
from utils import (DIS_METRICS_, EPOCHS, MAX_EPOCHS, eval_disen_xai, score_vae,
                   tf1_load, tf1_save)


class BetaVAE(MultiVAE):
    NAME = 'betavae'

def eval_betavae(trained_model, data, metrics, Ks, seed, run_path, classifier='gbt') -> dict:
    URM = data[ds_utils.TEST_IN]
    URM_test = data[ds_utils.TEST_OUT]

    results, _, train_codes = score_vae(run_path, BetaVAE.NAME, trained_model, URM, URM_test, metrics, Ks)

    if len(train_codes) > 0 and ds_utils.FACTORS_TRAIN in data and ds_utils.FACTORS_VALID in data:
        dis_metrics = [m for m in metrics if m in DIS_METRICS_]
        _, _, valid_codes = score_vae(run_path, BetaVAE.NAME, trained_model, data[ds_utils.VALID_TEST_DATA_IN], None,
                                  metrics, Ks)
        disen_xai_results = eval_disen_xai(dis_metrics, train_codes, data[ds_utils.FACTORS_TRAIN],
                                           valid_codes, data[ds_utils.FACTORS_VALID], seed, run_path, classifier)
        results.update(disen_xai_results)
    return results


def train_betavae(seed, path, data, metrics, Ks, early_stop_freq=1, early_stop_allow_worse=5, epochs=MAX_EPOCHS,
                  batch_size=500, layers_num=1, code_dim=200, reg=0.01, lr=1e-3, beta=1, keep=0.5):
    rng = np.random.default_rng(seed)
    tf1.reset_default_graph()
    tf1.set_random_seed(seed)

    train_data = data[ds_utils.TRAINING_DATA]

    n_users, n_items = train_data.shape
    idxlist = list(range(n_users))

    p_dims = [int(code_dim)]
    p_dims += [128 * np.power(2, i) for i in range(int(layers_num))]
    p_dims.append(n_items)

    betavae = BetaVAE(p_dims, None, reg, lr, seed)

    saver, logits_var, code_op, _, train_op_var = betavae.build_graph()

    config = tf1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf1.Session(config=config)

    model_out = {'sess': sess, 'output_op': logits_var, 'code_op': code_op, BetaVAE.NAME: betavae}

    saver_fn = partial(tf1_save, saver, sess, path)
    loader_fn = partial(tf1_load, saver, sess, path)
    eval_fn = partial(eval_betavae, model_out, data, metrics, Ks, seed, path)

    early_stop = EarlyStopper(eval_fn, saver_fn, loader_fn, early_stop_freq, early_stop_allow_worse)

    if 'evaluations' in path:
        # Skip training if a checkpoint exists in evaluation path
        try:
            loader_fn()
            return model_out
        except ValueError as e:
            pass

    init = tf1.global_variables_initializer()
    sess.run(init)

    for epoch in tqdm(range(1, epochs+1), desc=f'Training {path}', colour='blue', initial=1):
        rng.shuffle(idxlist)

        for st_idx in range(0, n_users, batch_size):
            end_idx = min(st_idx + batch_size, n_users)
            x = train_data[idxlist[st_idx: end_idx]]

            if sps.isspmatrix(x):
                x = x.toarray()
            x = x.astype('float32')

            sess.run(train_op_var, feed_dict={betavae.input_ph: x, betavae.keep_prob_ph: keep, betavae.anneal_ph: beta,
                                              betavae.is_training_ph: 1})
        
        # Early stopping
        if early_stop(epoch):
            break

    saver_fn()
    model_out.update({EPOCHS: epoch - early_stop_allow_worse * early_stop_freq if epoch < MAX_EPOCHS else MAX_EPOCHS})
    return model_out

