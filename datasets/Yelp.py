#!/usr/bin/env python3

import json
import os
import tarfile
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sps
from recpack.datasets.base import Dataset
from recpack.matrix.interaction_matrix import InteractionMatrix
from tqdm import tqdm


class Yelp(Dataset):
    USER_IX = 'user_id'
    ITEM_IX = 'business_id'
    RATING_IX = 'stars'

    DATASETURL = 'https://www.yelp.com/dataset/download'
    REMOTE_FILENAME = 'yelp_dataset'

    INTERACTIONS_FILENAME = 'yelp_academic_dataset_review.json'
    CATEGORIES_FILENAME = 'yelp_academic_dataset_business.json'

    @property
    def DEFAULT_FILENAME(self) -> str:
        return self.REMOTE_FILENAME

    def _download_dataset(self):
        if not os.path.exists(os.path.join(self.path, f'{self.REMOTE_FILENAME}.tar')):
            raise FileNotFoundError(f'Manually download the dataset from {self.DATASETURL} and place it under {self.path}')

    def _load_dataframe(self) -> pd.DataFrame:
        if not os.path.exists(os.path.join(self.path, self.INTERACTIONS_FILENAME)):
            self.fetch_dataset()

            # first untar the file
            with tarfile.open(os.path.join(self.path, f'{self.REMOTE_FILENAME}.tar')) as f:
                f.extract(self.INTERACTIONS_FILENAME, path=self.path)
        
        user_ids, item_ids, ratings = [], [], []
        with open(os.path.join(self.path, self.INTERACTIONS_FILENAME)) as f:
            for line in tqdm(f, desc=f'Parsing Yelp'):
                d = json.loads(line)
                user_ids.append(d[self.USER_IX])
                item_ids.append(d[self.ITEM_IX])
                ratings.append(int(d[self.RATING_IX]))
        df = pd.DataFrame({self.USER_IX: user_ids, self.ITEM_IX: item_ids, self.RATING_IX: ratings})
        df[self.ITEM_IX] = df[self.ITEM_IX]
        df[self.RATING_IX] = df[self.RATING_IX].astype(np.int8)
        return df
    
    def _load_business_categories(self) -> pd.DataFrame:
        if not os.path.exists(os.path.join(self.path, self.CATEGORIES_FILENAME)):
            with tarfile.open(os.path.join(self.path, f'{self.REMOTE_FILENAME}.tar')) as f:
                f.extract(self.CATEGORIES_FILENAME, path=self.path)

        bids, cates = [], []
        with open(os.path.join(self.path, self.CATEGORIES_FILENAME)) as f:
            for line in tqdm(f, desc='Parsing Yelp business categories'):
                d = json.loads(line)
                if 'categories' in d and d['categories'] is not None:
                    categories = [cat.strip() for cat in d['categories'].split(',')]
                    for c in categories:
                        bids.append(d[self.ITEM_IX])
                        cates.append(c)
        return pd.DataFrame({self.ITEM_IX: bids, 'categories': cates})
    
    def get_factors(self, interactions: pd.DataFrame) -> Tuple[pd.DataFrame, sps.spmatrix, dict, dict]:
        b2c_df = self._load_business_categories()

        # include only businesses that have interactions
        b2c_df = b2c_df[b2c_df[self.ITEM_IX].isin(interactions[self.ITEM_IX].unique())]

        categories_df = b2c_df.groupby('categories', as_index=False).agg({self.ITEM_IX: list})
        categories_df['count'] = categories_df[self.ITEM_IX].map(len)
        sorted_categories = categories_df.sort_values('count', ascending=False)
        top_categories = sorted_categories.head(100)

        # build business x category sparse matrix
        business_ids, categories = [], []
        for _, row in top_categories.iterrows():
            category = row['categories']
            bids = row[self.ITEM_IX]
            for b in bids:
                categories.append(category)
                business_ids.append(b)
        
        # use the same mapper as the interactions matrix
        bid2row = dict(interactions[[self.ITEM_IX, InteractionMatrix.ITEM_IX]].drop_duplicates().values)
        business_ids = list(map(bid2row.get, business_ids))

        cat2col = dict(zip(top_categories['categories'].unique(), range(len(top_categories))))
        categories = list(map(cat2col.get, categories))

        data = np.ones_like(categories)
        b2c_mat = sps.coo_matrix((data, (business_ids, categories)), shape=(max(business_ids) + 1, max(categories) + 1),
                                 dtype=np.int8)
        return pd.DataFrame({self.ITEM_IX: business_ids, 'category': categories}), b2c_mat, bid2row, cat2col
