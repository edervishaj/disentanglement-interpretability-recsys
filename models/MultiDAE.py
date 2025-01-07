#!/usr/bin/env python3

"""
Adapted from "Variational Autoencoders for Collaborative Filtering".
Official implementation: https://github.com/dawenl/vae_cf.
"""

from functools import partial

import numpy as np
import scipy.sparse as sps
import tensorflow._api.v2.compat.v1 as tf1
from tqdm import tqdm

import ds_utils
from models.helpers import EarlyStopper
from utils import (DIS_METRICS_, EPOCHS, MAX_EPOCHS, eval_disen_xai, score_vae,
                   tf1_load, tf1_save)


class MultiDAE:
    NAME = 'multidae'

    def __init__(self, p_dims, q_dims=None, lam=0.01, lr=1e-3, random_seed=None):
        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims
        self.dims = self.q_dims + self.p_dims[1:]
        
        self.lam = lam
        self.lr = lr
        self.random_seed = random_seed

        self.construct_weights()

        self.construct_placeholders()

    def construct_placeholders(self):
        self.input_ph = tf1.placeholder(dtype=tf1.float32, shape=[None, self.dims[0]])
        self.keep_prob_ph = tf1.placeholder_with_default(1.0, shape=None)

    def build_graph(self):
        saver, logits, code = self.forward_pass()
        log_softmax_var = tf1.nn.log_softmax(logits)

        # per-user average negative log-likelihood
        neg_ll = -tf1.reduce_mean(tf1.reduce_sum(log_softmax_var * self.input_ph, axis=1))
        # apply regularization to weights
        reg_var = tf1.add_n([tf1.nn.l2_loss(var) for var in self.weights])
        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        loss = neg_ll + 2 * self.lam * reg_var
        
        train_op = tf1.train.AdamOptimizer(self.lr).minimize(loss)

        return saver, logits, code, train_op

    def forward_pass(self):
        # construct forward graph
        h = tf1.nn.l2_normalize(self.input_ph, 1)
        h = tf1.nn.dropout(h, self.keep_prob_ph)
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = tf1.matmul(h, w) + b

            if i != len(self.weights) - 1:
                h = tf1.nn.tanh(h)

            if i == (len(self.weights) - 1)  // 2:
                z = h

        return tf1.train.Saver(), h, z

    def construct_weights(self):
        self.weights = []
        self.biases = []
        
        # define weights
        for i, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            weight_key = "weight_{}to{}".format(i, i+1)
            bias_key = "bias_{}".format(i+1)
            
            self.weights.append(tf1.get_variable(name=weight_key, shape=[d_in, d_out],
                                                 initializer=tf1.glorot_uniform_initializer(seed=self.random_seed)))
            
            self.biases.append(tf1.get_variable(name=bias_key, shape=[d_out],
                                                initializer=tf1.truncated_normal_initializer(stddev=0.001, seed=self.random_seed)))


def eval_multidae(trained_model, data, metrics, Ks, seed, run_path, classifier='gbt') -> dict:
    URM = data[ds_utils.TEST_IN]
    URM_test = data[ds_utils.TEST_OUT]

    results, _, train_codes = score_vae(run_path, MultiDAE.NAME, trained_model, URM, URM_test, metrics, Ks)

    if len(train_codes) > 0 and ds_utils.FACTORS_TRAIN in data and ds_utils.FACTORS_VALID in data:
        dis_metrics = [m for m in metrics if m in DIS_METRICS_]
        _, _, valid_codes = score_vae(run_path, MultiDAE.NAME, trained_model, data[ds_utils.VALID_TEST_DATA_IN], None,
                                      metrics, Ks)
        disen_xai_results = eval_disen_xai(dis_metrics, train_codes, data[ds_utils.FACTORS_TRAIN], valid_codes,
                                           data[ds_utils.FACTORS_VALID], seed, run_path, classifier)
        results.update(disen_xai_results)
    return results


def train_multidae(seed, path, data, metrics, Ks, early_stop_freq=1, early_stop_allow_worse=5,  epochs=MAX_EPOCHS, 
                   batch_size=500, layers_num=1, code_dim=200, reg=0.01, lr=1e-3):
    rng = np.random.default_rng(seed)
    tf1.reset_default_graph()
    tf1.set_random_seed(seed)

    train_data = data[ds_utils.TRAINING_DATA]

    n_users, n_items = train_data.shape
    idxlist = list(range(n_users))

    p_dims = [int(code_dim)]
    p_dims += [128 * np.power(2, i) for i in range(int(layers_num))]
    p_dims.append(n_items)

    multidae = MultiDAE(p_dims, None, reg, lr, seed)

    saver, logits_var, code_op, train_op_var = multidae.build_graph()

    config = tf1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf1.Session(config=config)

    model_out = {'sess': sess, 'output_op': logits_var, 'code_op': code_op, MultiDAE.NAME: multidae}

    saver_fn = partial(tf1_save, saver, sess, path)
    loader_fn = partial(tf1_load, saver, sess, path)
    eval_fn = partial(eval_multidae, model_out, data, metrics, Ks, seed, path)

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

            feed_dict = {multidae.input_ph: x, multidae.keep_prob_ph: 0.5}
            sess.run(train_op_var, feed_dict=feed_dict)
        
        # Early stopping
        if early_stop(epoch):
            break

    saver_fn()
    model_out.update({EPOCHS: epoch - early_stop_allow_worse * early_stop_freq if epoch < MAX_EPOCHS else MAX_EPOCHS})
    return model_out
