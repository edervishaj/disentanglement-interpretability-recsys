#!/usr/bin/env python3


import json
import os
import sys

import numpy as np
from recpack.datasets import Dataset
from recpack.matrix import InteractionMatrix
from recpack.scenarios import WeakGeneralization

from configurations import DATASETS
from datasets.BinarizeFilter import BinarizeFilter
from datasets.CoreFilter import CoreFilter
from utils import CLS, FACTORS_FN

DATASET_PATH = 'experiments/datasets/{}'
FULL_TRAINING_DATA = 'full_training_data'
TEST_DATA_IN = 'test_data_in'
TEST_DATA_OUT = 'test_data_out'
VALID_TRAINING_DATA = 'valid_train_data'
VALID_TEST_DATA_IN = 'valid_test_data_in'
VALID_TEST_DATA_OUT = 'valid_test_data_out'
GT_FACTORS_TRAIN = 'gt_factors_train'
GT_FACTORS_TEST = 'gt_factors_test'
GT_FACTORS_VALID_TRAIN = 'gt_factors_valid_train'
GT_FACTORS_VALID_TEST_OUT = 'gt_factors_valid_test'
GT_FACTORS_TRAIN_NPY = f'{GT_FACTORS_TRAIN}.npy'
GT_FACTORS_TEST_NPY = f'{GT_FACTORS_TEST}.npy'
GT_FACTORS_VALID_TRAIN_NPY = f'{GT_FACTORS_VALID_TRAIN}.npy'
GT_FACTORS_VALID_TEST_OUT_NPY = f'{GT_FACTORS_VALID_TEST_OUT}.npy'

TRAINING_DATA = 'training_data'
TEST_IN = 'test_in'
TEST_OUT = 'test_out'
FACTORS_TRAIN = 'factors_train'
FACTORS_TEST = 'factors_test'
FACTORS_VALID = 'factors_valid'


def data_path(dataset: str, seed: int, filename: str = '') -> str:
    return str(os.path.join(DATASET_PATH.format(dataset), str(seed), filename))


def make_dataset(d: str, min_user_ratings: int, min_item_ratings: int, min_rating_binarize: int, test_fraction: float,
                 validation_fraction: float, seed: int) -> None:
    """Downloads, saves and splits a dataset.
    
    Parameters
    ----------
    d : str
        The dataset name representing a key in `constants._datasets`.

    min_user_ratings : int
        The minimum number of ratings a user must have to be kept in the dataset.

    min_item_ratings : int
        The minimum number of ratings an item must have to be kept in the dataset.

    min_rating_binarize : int
        The minimum rating value for the binarization of the ratings. All ratings in
        [min_rating_binarize, *] are converted to 1 and everything (-inf, min_rating_binarize) set
        to 0.

    test_fraction : float
        The portion of users to keep as test set.

    validation_fraction : float
        The portion of users to consider as validation set.

    seed : int
        Seed for reproducible results.
    """

    dataset_parameters = dict(locals())
    config_path = data_path(d, seed, 'config.json')

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            saved_dataset_params = json.load(f)
            if {k: saved_dataset_params[k] for k in dataset_parameters} == dataset_parameters:
                return

    dataset_path = data_path(d, seed)

    if not os.path.isdir(dataset_path):
        os.makedirs(dataset_path)

    dataset: Dataset = DATASETS[d][CLS](path=dataset_path, use_default_filters=False)
    if min_rating_binarize >= 0:
        dataset.add_filter(BinarizeFilter(min_rating_binarize, dataset.RATING_IX))
    dataset.add_filter(CoreFilter(dataset.USER_IX, dataset.ITEM_IX, min_item_ratings, min_user_ratings))
    interactionMatrix = dataset.load()
    interactionMatrix.save(data_path(d, seed, d))

    dataset_parameters['num_interactions'] = int(interactionMatrix.values.nnz)
    dataset_parameters['active_users'] = int(interactionMatrix.num_active_users)
    dataset_parameters['active_items'] = int(interactionMatrix.num_active_items)
    dataset_parameters['sparsity'] = float(1.0 - interactionMatrix.density)
    dataset_parameters['minIPU'] = int(min(interactionMatrix.values.sum(axis=1).A1))
    dataset_parameters['minUPI'] = int(min(interactionMatrix.values.sum(axis=0).A1))

    # Random assignment of users into train-validation-test sets.
    tr_te_spl = WeakGeneralization(1.0 - test_fraction, validation=False, seed=seed)
    tr_val_spl = WeakGeneralization(1.0 - validation_fraction, validation=False, seed=seed)
        
    sets = []
    tr_te_spl.split(interactionMatrix)
    tr_te_spl.full_training_data.save(data_path(d, seed, FULL_TRAINING_DATA))
    tr_te_spl.test_data_in.save(data_path(d, seed, TEST_DATA_IN))
    tr_te_spl.test_data_out.save(data_path(d, seed, TEST_DATA_OUT))
    sets.append(tr_te_spl.test_data_in)
    if validation_fraction > 0.0:
        tr_val_spl.split(tr_te_spl.full_training_data)
        tr_val_spl.full_training_data.save(data_path(d, seed, VALID_TRAINING_DATA))
        tr_val_spl.test_data_in.save(data_path(d, seed, VALID_TEST_DATA_IN))
        tr_val_spl.test_data_out.save(data_path(d, seed, VALID_TEST_DATA_OUT))
        sets.append(tr_val_spl.test_data_in)

    # Prepare generative ground truth factors
    factors_fn = DATASETS[d].get(FACTORS_FN, None)
    if factors_fn is not None:
        Y_trains = factors_fn(dataset, interactionMatrix, sets, seed)

        gt_names = [GT_FACTORS_TRAIN, GT_FACTORS_VALID_TRAIN]
        gt_filenames = [GT_FACTORS_TRAIN_NPY, GT_FACTORS_VALID_TRAIN_NPY]

        # keep only features that have 2 classes in a binary assignment, with one at most representing 99% of the samples.
        for i, y_train in enumerate(Y_trains):
            ones_ratios = np.sum(y_train, axis=0) / y_train.shape[0]
            y_bin_cols = np.argwhere(np.logical_and(ones_ratios >= 0.01, ones_ratios < 0.99)).flatten()
            y_train = y_train[:, y_bin_cols]

            dataset_parameters[gt_names[i]] = y_train.shape[1]

            np.save(data_path(d, seed, gt_filenames[i]), y_train)
    else:
        print(f'Could not build generative factors for {d}', file=sys.stderr)

    # Save current dataset settings
    with open(config_path, 'w') as f:
        json.dump(dataset_parameters, f, indent=4)


def load_dataset(dataset: str, seed: int) -> dict:
    """Loads already prepared dataset for training/evaluation.

    Parameters
    ----------
    dataset : str
        The dataset name to load.
    
    seed : int
        The seed.

    Returns
    -------
    sets : dict
        Dictionary with keys consisting of FULL_TRAINING_DATA, TEST_DATA_IN, TEST_DATA_OUT and optionally of VALID_TRAIN_DATA, VALID_TEST_DATA_IN, VALID_TEST_DATA_OUT, GT_FACTORS_TRAIN, GT_FACTORS_TEST, GT_FACTORS_VALID_TRAIN, GT_FACTORS_VALID_TEST_OUT
    """
    sets = {} 
    full_training_data = InteractionMatrix.load(data_path(dataset, seed, FULL_TRAINING_DATA))
    sets[FULL_TRAINING_DATA] = full_training_data.values
    sets[TEST_DATA_IN] = InteractionMatrix.load(data_path(dataset, seed, TEST_DATA_IN)).values
    sets[TEST_DATA_OUT] = InteractionMatrix.load(data_path(dataset, seed, TEST_DATA_OUT)).values
    
    try:
        sets[GT_FACTORS_TRAIN] = np.load(data_path(dataset, seed, GT_FACTORS_TRAIN_NPY))
    except Exception:
        pass

    try:
        sets[VALID_TRAINING_DATA] = InteractionMatrix.load(data_path(dataset, seed, VALID_TRAINING_DATA)).values
        sets[VALID_TEST_DATA_IN] = InteractionMatrix.load(data_path(dataset, seed, VALID_TEST_DATA_IN)).values
        sets[VALID_TEST_DATA_OUT] = InteractionMatrix.load(data_path(dataset, seed, VALID_TEST_DATA_OUT)).values
        try:
            sets[GT_FACTORS_VALID_TRAIN] = np.load(data_path(dataset, seed, GT_FACTORS_VALID_TRAIN_NPY))
        except Exception:
            pass
    except Exception:
        pass
    return sets
