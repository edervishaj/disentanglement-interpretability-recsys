#!/usr/bin/env python3

"""
Adapted from "Learning Disentangled Representations for Recommendation".
Official implementation: https://jianxinma.github.io/disentangle-recsys.html.
"""

from functools import partial

import numpy as np
import scipy.sparse as sps
import tensorflow._api.v2.compat.v1 as tf1
from tensorflow_probability.python.distributions import RelaxedOneHotCategorical
from tqdm import tqdm

import ds_utils
from models.helpers import EarlyStopper
from utils import (DIS_METRICS_, EPOCHS, MAX_EPOCHS, eval_disen_xai, score_vae,
                   tf1_load, tf1_save)


class MacridVAE:
    NAME = 'macridvae'

    def __init__(self, seed, num_items, code_dim, dfac, rg, lr, tau, nogb, std):
        self.random_seed = seed
        self.lam = rg
        self.kfac = code_dim
        self.lr = lr
        self.tau = tau
        self.nogb = nogb
        self.std = std

        self.n_items = num_items

        # The first fc layer of the encoder Q is the context embedding table.
        self.q_dims = [num_items, dfac, dfac]
        self.weights_q, self.biases_q = [], []
        for i, (d_in, d_out) in enumerate(
                zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                d_out *= 2  # mu & var
            weight_key = "weight_q_{}to{}".format(i, i + 1)
            self.weights_q.append(tf1.get_variable(name=weight_key, shape=[d_in, d_out],
                                                   initializer=tf1.glorot_uniform_initializer(seed=self.random_seed)))
            bias_key = "bias_q_{}".format(i + 1)
            self.biases_q.append(tf1.get_variable(name=bias_key, shape=[d_out],
                                                  initializer=tf1.truncated_normal_initializer(stddev=0.001, seed=self.random_seed)))

        self.items = tf1.get_variable(name="items", shape=[num_items, dfac],
                                      initializer=tf1.glorot_uniform_initializer(seed=self.random_seed))

        self.cores = tf1.get_variable(name="cores", shape=[code_dim, dfac],
                                      initializer=tf1.glorot_uniform_initializer(seed=self.random_seed))

        self.input_ph = tf1.placeholder(dtype=tf1.float32, shape=[None, num_items])
        self.keep_prob_ph = tf1.placeholder_with_default(1., shape=None)
        self.is_training_ph = tf1.placeholder_with_default(0., shape=None)
        self.anneal_ph = tf1.placeholder_with_default(1., shape=None)

    def build_graph(self):
        saver, logits, facets_list, recon_loss, kl = self.forward_pass()

        reg_var = tf1.add_n([tf1.nn.l2_loss(var) for var in self.weights_q + [self.items, self.cores]])
        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        neg_elbo = recon_loss + self.anneal_ph * kl + 2. * self.lam * reg_var

        train_op = tf1.train.AdamOptimizer(self.lr).minimize(neg_elbo)
        return saver, logits, facets_list, train_op

    def q_graph_k(self, x):
        mu_q, std_q, kl = None, None, None
        h = tf1.nn.l2_normalize(x, 1)
        h = tf1.nn.dropout(h, self.keep_prob_ph)
        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            h = tf1.matmul(h, w, a_is_sparse=(i == 0)) + b
            if i != len(self.weights_q) - 1:
                h = tf1.nn.tanh(h)
            else:
                mu_q = h[:, :self.q_dims[-1]]
                mu_q = tf1.nn.l2_normalize(mu_q, axis=1)
                lnvarq_sub_lnvar0 = -h[:, self.q_dims[-1]:]
                std_q = tf1.exp(0.5 * lnvarq_sub_lnvar0) * self.std
                # Trick: KL is constant w.r.t. to mu_q after we normalize mu_q.
                kl = tf1.reduce_mean(tf1.reduce_sum(0.5 * (-lnvarq_sub_lnvar0 + tf1.exp(lnvarq_sub_lnvar0) - 1.), axis=1))
        return mu_q, std_q, kl

    def forward_pass(self):
        # clustering
        cores = tf1.nn.l2_normalize(self.cores, axis=1)
        items = tf1.nn.l2_normalize(self.items, axis=1)
        cates_logits = tf1.matmul(items, cores, transpose_b=True) / self.tau
        if self.nogb:
            cates = tf1.nn.softmax(cates_logits, axis=1)
        else:
            cates_dist = RelaxedOneHotCategorical(1, cates_logits)
            cates_sample = cates_dist.sample()
            cates_mode = tf1.nn.softmax(cates_logits, axis=1)
            cates = (self.is_training_ph * cates_sample + (1 - self.is_training_ph) * cates_mode)

        z_list = []
        probs, kl = None, None
        for k in range(int(self.kfac)):
            cates_k = tf1.reshape(cates[:, k], (1, -1))

            # q-network
            x_k = self.input_ph * cates_k
            mu_k, std_k, kl_k = self.q_graph_k(x_k)
            epsilon = tf1.random.normal(tf1.shape(std_k))
            z_k = mu_k + self.is_training_ph * epsilon * std_k
            kl = (kl_k if (kl is None) else (kl + kl_k))
            
            z_list.append(z_k)

            # p-network
            z_k = tf1.nn.l2_normalize(z_k, axis=1)
            logits_k = tf1.matmul(z_k, items, transpose_b=True) / self.tau
            probs_k = tf1.exp(logits_k)
            probs_k = probs_k * cates_k
            probs = (probs_k if (probs is None) else (probs + probs_k))

        logits = tf1.math.log(probs)
        logits = tf1.nn.log_softmax(logits)
        recon_loss = tf1.reduce_mean(tf1.reduce_sum(-logits * self.input_ph, axis=-1))

        code = tf1.transpose(tf1.stack(z_list, axis=0), perm=(1, 0, 2))

        return tf1.train.Saver(), logits, code, recon_loss, kl


def eval_macridvae(trained_model, data, metrics, Ks, seed, run_path, classifier='gbt') -> dict:
    URM = data[ds_utils.TEST_IN]
    URM_test = data[ds_utils.TEST_OUT]

    results, _, train_codes = score_vae(run_path, MacridVAE.NAME, trained_model, URM, URM_test, metrics, Ks)
    train_codes = np.mean(train_codes, axis=-1)

    if len(train_codes) > 0 and ds_utils.FACTORS_TRAIN in data and ds_utils.FACTORS_VALID in data:
        dis_metrics = [m for m in metrics if m in DIS_METRICS_]
        _, _, valid_codes = score_vae(run_path, MacridVAE.NAME, trained_model, data[ds_utils.VALID_TEST_DATA_IN], None,
                                      metrics, Ks)
        valid_codes = np.mean(valid_codes, axis=-1)
        disen_xai_results = eval_disen_xai(dis_metrics, train_codes, data[ds_utils.FACTORS_TRAIN],
                                           valid_codes, data[ds_utils.FACTORS_VALID], seed, run_path, classifier)
        results.update(disen_xai_results)
    return results

def train_macridvae(seed, path, data, metrics, Ks, early_stop_freq=1, early_stop_allow_worse=5, epochs=MAX_EPOCHS,
                    batch_size=500, code_dim=7, dfac=100, reg=0.0, lr=1e-3, tau=0.1, nogb=False, std=0.075, beta=0.2, keep=0.5):
    rng = np.random.default_rng(seed)
    tf1.reset_default_graph()
    tf1.set_random_seed(seed)

    train_data = data[ds_utils.TRAINING_DATA]

    n_users, n_items = train_data.shape
    idxlist = list(range(n_users))

    num_batches = int(np.ceil(float(n_users) / batch_size))
    total_anneal_steps = 5 * num_batches

    vae = MacridVAE(seed, n_items, int(code_dim), dfac, reg, lr, tau, nogb, std)
    saver, logits_var, z_hat, train_op_var = vae.build_graph()

    config = tf1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf1.Session(config=config)

    model_out = {'sess': sess, 'output_op': logits_var, 'code_op': z_hat, MacridVAE.NAME: vae}

    saver_fn = partial(tf1_save, saver, sess, path)
    loader_fn = partial(tf1_load, saver, sess, path)
    eval_fn = partial(eval_macridvae, model_out, data, metrics, Ks, seed, path)
    
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
            x = train_data[idxlist[st_idx:end_idx]]
            if sps.isspmatrix(x):
                x = x.toarray()
            x = x.astype('float32')
            if total_anneal_steps > 0:
                anneal = min(beta, 1. * update_count / total_anneal_steps)
            else:
                anneal = beta
            feed_dict = {vae.input_ph: x, vae.keep_prob_ph: keep, vae.anneal_ph: anneal, vae.is_training_ph: 1}
            sess.run(train_op_var, feed_dict=feed_dict)
            update_count += 1

        # Early stopping
        if early_stop(epoch):
            break
    
    saver_fn()
    model_out.update({EPOCHS: epoch - early_stop_allow_worse * early_stop_freq if epoch < MAX_EPOCHS else MAX_EPOCHS})
    return model_out
