#!/usr/bin/env python3


import argparse
import os
import pickle
import sys
import time
import warnings
from datetime import timedelta

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=Warning)

from functools import partial

import tensorflow._api.v2.compat.v1 as tf1

tf1.disable_v2_behavior()

from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, tpe

import ds_utils
from configurations import COMMON_PARAM_SPACE, DATASETS, RECS
from utils import (DIS_METRICS_, EPOCHS, EVAL_FN, MAX_EPOCHS, PARAM_SPACE,
                   REL_METRICS_, TRAIN_FN, clean_path)

TRAIN_TIME = 'train_time'
EVAL_TIME = 'test_time'
PARAMS = 'params'
PARAM_SEARCH = 'param-search'
EVALUTION = 'evaluations'


def obj_fun(train_kwargs: dict, path: str, seed: int, Ks: list, model: str, data: dict):
    """Objective function to be utilized by Hyperopt fmin(). Trains a model and returns metric that
    is optimized.

    Parameters
    ----------

    train_kwargs : dict
        Dictionary containing the hyperparameters selected by Hyperopt fmin().

    path : str
        Path where to save/load the best model whilst training.

    seed : int
        Seed for reproducibility.

    metrics : list[str]
        List of metrics to evaluate the trained model with.

    weight: list[float]
        The list of weights for metrics in case of multi-stake tuning.

    Ks : list[int]
        List of cutoffs to evaluate the trained model with.

    model : str
        The model ID.

    data : dict
        The datasets to be used for training.

    Returns
    -------

    dict
        A dictionary with at least 'loss', 'status' keys as required by Hyperopt fmin().
    """

    # Clean path
    clean_path(path)

    # train with hyperparams
    try:
        start_train = time.time()
        model_trained = RECS[model][TRAIN_FN](seed, path, data, ['ndcg'], Ks, early_stop_freq=5, 
                                              early_stop_allow_worse=2, **train_kwargs)
        end_train = time.time()

        if train_kwargs[EPOCHS] - model_trained[EPOCHS] > 2:
            train_kwargs[EPOCHS] = model_trained[EPOCHS]

        del model_trained[EPOCHS]

        # Check model performance
        start_eval = time.time()
        results = RECS[model][EVAL_FN](model_trained, data, ['ndcg'], Ks, seed, path)
        end_eval = time.time()
    except tf1.errors.ResourceExhaustedError as e:
        return {
            'loss': sys.maxsize,
            'status': STATUS_FAIL,
            TRAIN_TIME: end_train - start_train,
            PARAMS: train_kwargs,
            EVAL_TIME: end_eval - start_eval
        }

    return {
        'loss': -list(results.values())[0],
        'status': STATUS_OK,
        PARAMS: train_kwargs,
        EVAL_TIME: end_eval - start_eval,
        TRAIN_TIME: end_train - start_train
    }


def parameter_search(model: str, dataset: str, evals: int, K: int, seed: int):
    """Performs hyperparameter search.

    Parameters
    ----------

    model : str
        The model name to evaluate.

    dataset : str
        The dataset name to load for evaluation.

    evals : int
        Number of Bayesian Search Optimization evaluations.

    K : int
        The cutoff for the metric.

    seed : int
        Seed for reproducibility.
    """

    start_time = time.time()

    data = ds_utils.load_dataset(dataset, seed)
    if len(data) == 2:
        raise ValueError(f'{dataset} has been prepared without a validation set!')
    
    sets = {
        ds_utils.TRAINING_DATA: data[ds_utils.VALID_TRAINING_DATA],
        ds_utils.TEST_IN: data[ds_utils.VALID_TEST_DATA_IN],
        ds_utils.TEST_OUT: data[ds_utils.VALID_TEST_DATA_OUT],
        ds_utils.VALID_TEST_DATA_IN: data[ds_utils.VALID_TEST_DATA_IN]
    }

    if ds_utils.GT_FACTORS_TRAIN in data and ds_utils.GT_FACTORS_VALID_TRAIN in data:
        sets[ds_utils.FACTORS_TRAIN] = data[ds_utils.GT_FACTORS_TRAIN]
        sets[ds_utils.FACTORS_VALID] = data[ds_utils.GT_FACTORS_VALID_TRAIN]

    params = dict(COMMON_PARAM_SPACE)
    params.update(RECS[model][PARAM_SPACE])
    
    run_name = f'{model}-{dataset}-{seed}-{evals}'
    run_path = os.path.join('experiments', PARAM_SEARCH, run_name)

    if not os.path.isdir(run_path):
        os.makedirs(run_path)

    trials = Trials()
    trials_path = str(os.path.join(f'{run_path}.hyperopt')) 

    fmin(
        fn=partial(obj_fun, path=run_path, seed=seed, Ks=[K], model=model, data=sets),
        space=params,
        algo=tpe.suggest,
        max_evals=evals,
        trials=trials,
        rstate=np.random.default_rng(seed)
    )

    with open(trials_path, 'wb') as f:
        pickle.dump(trials, f, pickle.HIGHEST_PROTOCOL)

    end_time = time.time()

    with open(os.path.join(run_path, 'time.txt'), 'w') as f:
        f.write(str(timedelta(seconds=end_time - start_time)))


def evaluate(model: str, dataset: str, metrics: list, K: list, evals: int, seed: int, classifier: str) -> None:
    """Evaluates a model on a specific dataset through a list of metrics and cutoffs.

    Parameters
    ----------

    model : str
        The model name to evaluate.

    dataset : str
        The dataset name to load for evaluation.

    metrics : list[str]
        The list of metrics to evaluate the model on. The model is evaluated on a cross product of metrics with K.

    K : list[int]
        The cutoffs for the metrics. The model is evaluated on a cross product of metrics with K.

    evals : int
        Number of Bayesian Search Optimization evaluations.

    seed : int
        Seed for reproducibility.

    classifier : str
        Model type to be used for disentanglement/interpretability metrics.
    """
    
    start_time = time.time()

    run_name = f'{model}-{dataset}-{seed}-{evals}'
    trials_path = os.path.join('experiments', PARAM_SEARCH, f'{run_name}.hyperopt')

    eval_path = os.path.join('experiments', EVALUTION, run_name)
    if not os.path.isdir(eval_path):
        os.makedirs(eval_path)

    data = ds_utils.load_dataset(dataset, seed)

    sets = {
        ds_utils.TRAINING_DATA: data[ds_utils.FULL_TRAINING_DATA],
        ds_utils.TEST_IN: data[ds_utils.TEST_DATA_IN],
        ds_utils.TEST_OUT: data[ds_utils.TEST_DATA_OUT],
        ds_utils.VALID_TEST_DATA_IN: data[ds_utils.VALID_TEST_DATA_IN]
    }

    if ds_utils.GT_FACTORS_TRAIN not in data:
        for m in DIS_METRICS_:
            metrics.remove(m)

    if ds_utils.GT_FACTORS_TRAIN in data:
        sets[ds_utils.FACTORS_TRAIN] = data[ds_utils.GT_FACTORS_TRAIN]
        sets[ds_utils.FACTORS_VALID] = data[ds_utils.GT_FACTORS_VALID_TRAIN]

    # access trials
    with open(trials_path, 'rb') as f:
        trials = pickle.load(f)

    best_trial = trials.best_trial
    params = best_trial['result'][PARAMS]

    # train model with best params
    trained_model = RECS[model][TRAIN_FN](seed, eval_path, sets, metrics, K, early_stop_freq=MAX_EPOCHS+1, **params)

    # evaluate
    results = RECS[model][EVAL_FN](trained_model, sets, metrics, K, seed, eval_path, classifier)

    for key in results:
        with open(os.path.join(eval_path, f'{key}.txt'), 'w') as f:
            f.write(str(results[key]))

    end_time = time.time()

    with open(os.path.join(eval_path, 'time.txt'), 'w') as f:
        f.write(str(timedelta(seconds=end_time - start_time)))


def main(args) -> None:
    """Runs command with respective arguments."""
    if args.command == 'prep-dataset':
        ds_utils.make_dataset(args.dataset, args.min_user_ratings, args.min_item_ratings,
                        args.min_rating_binarize, args.test_fraction, args.validation_fraction, args.seed)
    elif args.command == 'param-search':
        parameter_search(args.model, args.dataset, args.evals, args.K, args.seed)
    elif args.command == 'eval':
        evaluate(args.model, args.dataset, args.metrics, args.K, args.evals, args.seed, args.classifier)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(title='commands', dest='command')

    ################
    # prep-dataset #
    ################
    dataset_parser = subparsers.add_parser('prep-dataset', formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                           help='download and split dataset')
    dataset_parser.add_argument('dataset', help='dataset to prepare', type=str, choices=list(DATASETS.keys()))
    dataset_parser.add_argument('--min-user-ratings', type=int, default=1,
                                help='minimum number of ratings a user must have')
    dataset_parser.add_argument('--min-item-ratings', type=int, default=1,
                                help='minimum number of ratings an item must have')
    dataset_parser.add_argument('--min-rating-binarize', type=float, default=-1,
                                help='the minimum rating value to set to 1 with lower values set to 0')
    dataset_parser.add_argument('--test-fraction', type=float, default=0.2,
                                help='test set fraction in a random train-test split')
    dataset_parser.add_argument('--validation-fraction', type=float, default=0.0,
                                help='validation set fraction in a random train-validation-test split')
    dataset_parser.add_argument('--seed', type=int, default=41)

    
    ################
    # param-search #
    ################
    search_parser = subparsers.add_parser('param-search', help='run hyperparamenter search',
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    search_parser.add_argument('model', type=str, choices=list(RECS.keys()), help='recommendation model to optimize')
    search_parser.add_argument('dataset', type=str, choices=list(DATASETS.keys()),
                               help='dataset to search parameters for')
    search_parser.add_argument('--evals', type=int, default=50, help='number of bayesian optimization evaluations')
    search_parser.add_argument('--K', type=int, default=100, help='cutoff value for evaluation')
    search_parser.add_argument('--seed', type=int, default=41)


    ########
    # eval #
    ########
    eval_parser = subparsers.add_parser('eval', help='evaluate model on dataset',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    eval_parser.add_argument('model', type=str, choices=list(RECS.keys()))
    eval_parser.add_argument('dataset', type=str, choices=list(DATASETS.keys()))
    eval_parser.add_argument('--metrics', nargs='*', type=str, default=list(REL_METRICS_) + list(DIS_METRICS_.keys()))
    eval_parser.add_argument('--K', nargs='*', type=int, default=[100], help='cutoff value for evaluation')
    eval_parser.add_argument('--evals', help='Number of BSO evaluations performed', type=int, default=50)
    eval_parser.add_argument('--seed', help='seed', type=int, default=41)
    eval_parser.add_argument('--classifier', type=str, choices=['logistic', 'rf', 'gbt'],
                             default='gbt', help='type of model for disentanglement/interpretability metrics')
    
    args = parser.parse_args()

    main(args)
