#!/usr/bin/env python3

from hyperopt import hp
from recpack.datasets import MovieLens1M

from datasets.AmazonCD import AmazonCD
from datasets.GoodReadsChildren import GoodReadsChildren
from datasets.Yelp import Yelp
from generative_factors import (GoodReads_shelves, ML1M_tags,
                                amazon_cd_categories, yelp_categories)
from models.BetaVAE import eval_betavae, train_betavae
from models.MacridVAE import eval_macridvae, train_macridvae
from models.MultiDAE import eval_multidae, train_multidae
from models.MultiVAE import eval_multivae, train_multivae
from models.PureSVD import eval_puresvd, train_puresvd
from models.TopPop import eval_toppop, train_toppop
from utils import CLS, EVAL_FN, FACTORS_FN, MAX_EPOCHS, PARAM_SPACE, TRAIN_FN


COMMON_PARAM_SPACE = {
    'epochs': hp.choice('epochs', [MAX_EPOCHS]),
    'batch_size': hp.choice('batch_size', [128, 256, 512, 1024]),
    'lr': hp.loguniform('lr', -10, -2),
    'code_dim': hp.quniform('code_dim', 2, 20, 1)
}


DATASETS = {
    'GoodReadsChildren': {
        CLS: GoodReadsChildren,
        FACTORS_FN: GoodReads_shelves
    },
    'ML1M': {
        CLS: MovieLens1M,
        FACTORS_FN: ML1M_tags
    },
    'Yelp': {
        CLS: Yelp,
        FACTORS_FN: yelp_categories
    },
    'AmazonCD': {
        CLS: AmazonCD,
        FACTORS_FN: amazon_cd_categories
    }
}

RECS = {
    'multidae': {
        TRAIN_FN: train_multidae,
        EVAL_FN: eval_multidae,
        PARAM_SPACE: {
            'reg': hp.loguniform('reg', -10, 0),
            'layers_num': hp.quniform('layers_num', 0, 4, 1),
        }
    },
    'multivae': {
        TRAIN_FN: train_multivae,
        EVAL_FN: eval_multivae,
        PARAM_SPACE: {
            'reg': hp.loguniform('reg', -10, 0),
            'layers_num': hp.quniform('layers_num', 0, 4, 1),
            'beta': hp.quniform('beta', 1, 100, 1) * 0.01,
        }
    },
    'betavae': {
        TRAIN_FN: train_betavae,
        EVAL_FN: eval_betavae,
        PARAM_SPACE: {
            'reg': hp.loguniform('reg', -12, 0),
            'layers_num': hp.quniform('layers_num', 0, 4, 1),
            'beta': hp.loguniform('beta', 0.1, 6),
        }
    },
    'macridvae': {
        TRAIN_FN: train_macridvae,
        EVAL_FN: eval_macridvae,
        PARAM_SPACE: {
            'reg': hp.loguniform('reg', -12, 0),
            'keep': hp.quniform('dropout', 1, 20, 1) * 0.05,
            'beta': hp.quniform('beta', 1, 100, 1) * 0.05,
            # 'tau': hp.quniform('tau', 1, 20, 1) * 0.05,       # set to 0.1 in the paper
            'std': hp.quniform('std', 1, 10, 1) * 0.025
        }
    },
    'toppop': {
        TRAIN_FN: train_toppop,
        EVAL_FN: eval_toppop,
        PARAM_SPACE: {}
    },
    'puresvd': {
        TRAIN_FN: train_puresvd,
        EVAL_FN: eval_puresvd,
        PARAM_SPACE: {
            'epochs': hp.choice('epochs', [MAX_EPOCHS]),
            'batch_size': hp.choice('batch_size', [128]),
            'lr': hp.choice('lr', [1e-1]),
        }
    }
}
