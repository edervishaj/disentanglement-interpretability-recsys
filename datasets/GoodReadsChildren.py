#!/usr/bin/env python3

import gzip
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from datasets.GoodReads import GoodReads


class GoodReadsChildren(GoodReads):
    DATASETURL = 'https://drive.google.com/uc?id=1Cf90P5TH84ufrs8qyLrM-iWOXOGjBi9r'
    REMOTE_FILENAME = 'goodreads_interactions_children.json.gz'
    
    SHELVESURL = 'https://drive.google.com/uc?id=1R3wJPgyzEX9w6EI8_LmqLbpY4cIC9gw4'
    SHELVES_FILENAME = 'goodreads_books_children.json.gz'

    # Merged bookshelves
    GROUPING_DICT = {
        'childrens': 'children',
        'children': 'children',
        'children-s-books': 'children',
        'children-s': 'children',
        'kids': 'children',
        'children-books': 'children',
        'childrens-books': 'children',
        'kids-books': 'children',
        'children-s-literature': 'children',
        'children-s-lit': 'children',
        'childhood': 'children',
        'childrens-lit': 'children',
        'childhood-books': 'children',
        'childrens-literature': 'children',
        'childhood-reads': 'children',
        'children-s-book': 'children',
        'kid-books': 'children',
        'kid-lit': 'children',
        'owned': 'owned',
        'books-i-own': 'owned',
        'owned-books': 'owned',
        'my-books': 'owned',
        'i-own': 'owned',
        'own-it': 'owned',
        'picture-book': 'picture-books',
        'picture-books': 'picture-books',
        'picturebooks': 'picture-books',
        'children-s-picture-books': 'picture-books',
        'childrens-picture-books': 'picture-books',
        'picture': 'picture-books',
        'young-adult': 'youth',
        'youth': 'youth',
        'library': 'library',
        'my-library': 'library',
        'fiction': 'fiction',
        'children-s-fiction': 'fiction',
        'childrens-fiction': 'fiction',
        'realistic-fiction': 'fiction',
        'school': 'school',
        'pre-school': 'school',
        'preschool': 'school',
        'classic': 'classic',
        'classics': 'classic',
    }

    # Dropped bookshelves
    DROPPED = [
        'to-read',
        'currently-reading',
        'children',
        'favorites',
        'owned',
        'default',
        'library',
        'picture-books',
        'ya',
        'read-aloud',
        'to-buy',
        'read-in-2016',
        'childhood-favorites',
        'kindle',
        'books',
        'read-alouds',
        'chapter-books',
        'favourites',
        'read-in-2015',
        'read-in-2017',
        'wish-list',
        'ebook',
        'classroom-library',
    ]

    POST_PROCESS = ['juvenile-fiction']

    def _load_dataframe(self) -> pd.DataFrame:
        self.fetch_dataset()
        user_ids, item_ids, ratings = [], [], []
        with gzip.open(os.path.join(self.path, self.DEFAULT_FILENAME), 'rb') as f:
            for line in tqdm(f, desc=f'Parsing {__class__.__name__}'):
                d = json.loads(line)
                user_ids.append(d[self.USER_IX])
                item_ids.append(int(d[self.ITEM_IX]))
                ratings.append(int(d[self.RATING_IX]))
        df = pd.DataFrame({self.USER_IX: user_ids, self.ITEM_IX: item_ids, self.RATING_IX: ratings})
        df[self.ITEM_IX] = df[self.ITEM_IX].astype(np.int32)
        df[self.RATING_IX] = df[self.RATING_IX].astype(np.int8)
        return df
