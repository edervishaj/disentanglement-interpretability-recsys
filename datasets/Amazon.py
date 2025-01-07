#!/usr/bin/env python3

import gzip
import json
import os
from typing import Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import scipy.sparse as sps
from recpack.datasets.base import Dataset
from recpack.matrix.interaction_matrix import InteractionMatrix
from tqdm import tqdm


def _fetch_remote(url, filename):
    pbar = None

    def _show_progress(block_num, block_size, total_size):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=f'Downloading {filename}')
        downloaded = block_num * block_size
        if downloaded < total_size:
            pbar.update(block_size)
        else:
            pbar.close()
            pbar = None

    urlretrieve(url, filename, _show_progress)


class Amazon(Dataset):
    """
    Base class for Amazon datasets
    """
    USER_IX = 'reviewerID'
    ITEM_IX = 'asin'
    RATING_IX = 'overall'

    DATASETURL = ''
    REMOTE_FILENAME = ''

    METADATAURL = ''
    METADATA_FILENAME = ''

    CAT_TO_DROP = []

    @property
    def DEFAULT_FILENAME(self) -> str:
        return self.REMOTE_FILENAME

    def _load_dataframe(self) -> pd.DataFrame:
        filename_path = os.path.join(self.path, self.DEFAULT_FILENAME)
        if not os.path.exists(filename_path):
            _fetch_remote(self.DATASETURL, filename_path)

        user_ids, item_ids, ratings = [], [], []
        with gzip.open(filename_path) as f:
            for line in tqdm(f, desc=f'Parsing {self.__class__.__name__}'):
                d = json.loads(line)
                user_ids.append(d[self.USER_IX])
                item_ids.append(d[self.ITEM_IX])
                ratings.append(float(d[self.RATING_IX]))
        return pd.DataFrame({self.USER_IX: user_ids, self.ITEM_IX: item_ids, self.RATING_IX: ratings})

    def _load_categories(self) -> pd.DataFrame:
        metadata_file = os.path.join(self.path, self.METADATA_FILENAME)
        if not os.path.exists(metadata_file):
            _fetch_remote(self.METADATAURL, metadata_file)

        items, categories = [], []
        with gzip.open(metadata_file, 'rb') as f:
            for line in tqdm(f, desc=f'Parsing {self.__class__.__name__} metadata'):
                d = json.loads(line)
                cats = d['category']
                for c in self.CAT_TO_DROP:
                    if c in cats:
                        cats.remove(c)
                categories.extend(cats)
                items.extend([d[self.ITEM_IX]] * len(cats))
        return pd.DataFrame({self.ITEM_IX: items, 'category': categories})

    def get_factors(self, interactions: pd.DataFrame) -> Tuple[pd.DataFrame, sps.spmatrix, dict, dict]:
        item2cat_df = self._load_categories()

        # include only CDs that have interactions
        item2cat_df = item2cat_df[item2cat_df[self.ITEM_IX].isin(interactions[self.ITEM_IX].unique())]

        categories_df = item2cat_df.groupby('category', as_index=False).agg({self.ITEM_IX: list})
        categories_df['count'] = categories_df[self.ITEM_IX].map(len)
        sorted_categories = categories_df.sort_values('count', ascending=False)
        top_categories = sorted_categories.head(100)

        # build CD x category sparse matrix
        item_ids, categories = [], []
        for _, row in top_categories.iterrows():
            category = row['category']
            cdids = row[self.ITEM_IX]
            for c in cdids:
                categories.append(category)
                item_ids.append(c)

        inter_grouped = interactions.groupby(self.ITEM_IX, as_index=False).agg({InteractionMatrix.ITEM_IX: 'first'})
        itemid2row = dict(zip(inter_grouped[self.ITEM_IX], inter_grouped[InteractionMatrix.ITEM_IX]))
        item_ids = list(map(itemid2row.get, item_ids))

        cat2col = dict(zip(top_categories['category'].unique(), range(len(top_categories))))
        categories = list(map(cat2col.get, categories))

        data = np.ones_like(categories)
        item2cat_mat = sps.coo_matrix((data, (item_ids, categories)), shape=(max(item_ids) + 1, max(categories) + 1),
                                      dtype=np.int8)
        return pd.DataFrame({self.ITEM_IX: item_ids, 'category': categories}), item2cat_mat, itemid2row, cat2col
