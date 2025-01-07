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
from models.MultiDAE import MultiDAE
from utils import (DIS_METRICS_, EPOCHS, MAX_EPOCHS, eval_disen_xai, score_vae,
                   tf1_load, tf1_save)


class MultiVAE(MultiDAE):
    NAME = 'multivae'

    def construct_placeholders(self):
        super().construct_placeholders()

        # placeholders with default values when scoring
        self.is_training_ph = tf1.placeholder_with_default(0., shape=None)
        self.anneal_ph = tf1.placeholder_with_default(1., shape=None)

    def build_graph(self):
        saver, logits, code, KL = self.forward_pass()
        log_softmax_var = tf1.nn.log_softmax(logits)

        neg_ll = -tf1.reduce_mean(tf1.reduce_sum(log_softmax_var * self.input_ph, axis=-1))
        # apply regularization to weights
        reg_var = tf1.add_n([tf1.nn.l2_loss(var) for var in self.weights_q + self.weights_p])
        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        neg_ELBO = neg_ll + self.anneal_ph * KL + 2 * self.lam * reg_var
        
        train_op = tf1.train.AdamOptimizer(self.lr).minimize(neg_ELBO)

        return saver, logits, code, neg_ELBO, train_op
    
    def q_graph(self):
        mu_q, std_q, KL = None, None, None
        
        h = tf1.nn.l2_normalize(self.input_ph, 1)
        h = tf1.nn.dropout(h, self.keep_prob_ph)
        
        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            h = tf1.matmul(h, w) + b
            
            if i != len(self.weights_q) - 1:
                h = tf1.nn.tanh(h)
            else:
                mu_q = h[:, :self.q_dims[-1]]
                logvar_q = h[:, self.q_dims[-1]:]

                std_q = tf1.exp(0.5 * logvar_q)
                KL = tf1.reduce_mean(tf1.reduce_sum(0.5 * (-logvar_q + tf1.exp(logvar_q) + mu_q**2 - 1), axis=1))
        return mu_q, std_q, KL
    
    def p_graph(self, z):
        h = z
        
        for i, (w, b) in enumerate(zip(self.weights_p, self.biases_p)):
            h = tf1.matmul(h, w) + b
            
            if i != len(self.weights_p) - 1:
                h = tf1.nn.tanh(h)
        return h

    def forward_pass(self):
        # q-network
        mu_q, std_q, KL = self.q_graph()

        epsilon = tf1.random_normal(tf1.shape(std_q))

        sampled_z = mu_q + self.is_training_ph * epsilon * std_q

        # p-network
        logits = self.p_graph(sampled_z)

        return tf1.train.Saver(), logits, sampled_z, KL
    
    def construct_weights(self):
        self.weights_q, self.biases_q = [], []
        
        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                # we need two sets of parameters for mean and variance,
                # respectively
                d_out *= 2
            weight_key = "weight_q_{}to{}".format(i, i+1)
            bias_key = "bias_q_{}".format(i+1)
            
            self.weights_q.append(tf1.get_variable(name=weight_key, shape=[d_in, d_out],
                                                   initializer=tf1.glorot_uniform_initializer(seed=self.random_seed)))
            
            self.biases_q.append(tf1.get_variable(name=bias_key, shape=[d_out],
                                                  initializer=tf1.truncated_normal_initializer(stddev=0.001, seed=self.random_seed)))
            
        self.weights_p, self.biases_p = [], []

        for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
            weight_key = "weight_p_{}to{}".format(i, i+1)
            bias_key = "bias_p_{}".format(i+1)
            self.weights_p.append(tf1.get_variable(name=weight_key, shape=[d_in, d_out],
                                                   initializer=tf1.glorot_uniform_initializer(seed=self.random_seed)))
            
            self.biases_p.append(tf1.get_variable(name=bias_key, shape=[d_out],
                                                  initializer=tf1.truncated_normal_initializer(stddev=0.001, seed=self.random_seed)))


def eval_multivae(trained_model, data, metrics, Ks, seed, run_path, classifier='gbt') -> dict:
    URM = data[ds_utils.TEST_IN]
    URM_test = data[ds_utils.TEST_OUT]

    results, _, train_codes = score_vae(run_path, MultiVAE.NAME, trained_model, URM, URM_test, metrics, Ks)

    if len(train_codes) > 0 and ds_utils.FACTORS_TRAIN in data and ds_utils.FACTORS_VALID in data:
        dis_metrics = [m for m in metrics if m in DIS_METRICS_]
        _, _, valid_codes = score_vae(run_path, MultiVAE.NAME, trained_model, data[ds_utils.VALID_TEST_DATA_IN], None,
                                  metrics, Ks)
        disen_xai_results = eval_disen_xai(dis_metrics, train_codes, data[ds_utils.FACTORS_TRAIN],
                                           valid_codes, data[ds_utils.FACTORS_VALID], seed, run_path, classifier)
        results.update(disen_xai_results)
    return results


def train_multivae(seed, path, data, metrics, Ks, early_stop_freq=1, early_stop_allow_worse=5, epochs=MAX_EPOCHS,
                   batch_size=500, layers_num=1, code_dim=200, reg=0.01, lr=1e-3, beta=0.2, keep=0.5):
    rng = np.random.default_rng(seed)
    tf1.compat.v1.reset_default_graph()
    tf1.compat.v1.set_random_seed(seed)

    train_data = data[ds_utils.TRAINING_DATA]

    n_users, n_items = train_data.shape
    idxlist = list(range(n_users))

    p_dims = [int(code_dim)]
    p_dims += [128 * np.power(2, i) for i in range(int(layers_num))]
    p_dims.append(n_items)

    total_anneal_steps = 200000

    multivae = MultiVAE(p_dims, None, reg, lr, seed)

    saver, logits_var, code, _, train_op_var = multivae.build_graph()

    config = tf1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf1.Session(config=config)

    model_out = {'sess': sess, 'output_op': logits_var, 'code_op': code, MultiVAE.NAME: multivae}

    saver_fn = partial(tf1_save, saver, sess, path)
    loader_fn = partial(tf1_load, saver, sess, path)
    eval_fn = partial(eval_multivae, model_out, data, metrics, Ks, seed, path)

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

    update_count = 0.0

    for epoch in tqdm(range(1, epochs+1), desc=f'Training {path}', colour='blue', initial=1):
        rng.shuffle(idxlist)

        for st_idx in range(0, n_users, batch_size):
            end_idx = min(st_idx + batch_size, n_users)
            x = train_data[idxlist[st_idx: end_idx]]

            if sps.isspmatrix(x):
                x = x.toarray()
            x = x.astype('float32')

            if total_anneal_steps > 0:
                anneal = min(beta, 1. * update_count / total_anneal_steps)
            else:
                anneal = beta

            sess.run(train_op_var, feed_dict={multivae.input_ph: x, multivae.keep_prob_ph: keep,
                                              multivae.anneal_ph: anneal, multivae.is_training_ph: 1})

            update_count += 1
        
        # Early stopping
        if early_stop(epoch):
            break

    saver_fn()
    model_out.update({EPOCHS: epoch - early_stop_allow_worse * early_stop_freq if epoch < MAX_EPOCHS else MAX_EPOCHS})
    return model_out
