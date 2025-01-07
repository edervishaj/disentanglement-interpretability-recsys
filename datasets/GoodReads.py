#!/usr/bin/env python3

import gzip
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sps
from recpack.datasets import Dataset
from tqdm import tqdm


class GoodReads(Dataset):
    """
    Base class for GoodReads datasets
    """

    USER_IX = 'user_id'
    ITEM_IX = 'book_id'
    RATING_IX = 'rating'

    DATASETURL = 'https://drive.google.com/uc?id=1zmylV7XW2dfQVCLeg1LbllfQtHD2KUon'
    REMOTE_FILENAME = 'goodreads_interactions.csv'

    SHELVESURL = 'https://drive.google.com/uc?id=1LXpK1UfqtP89H1tYy0pBGHjYk8IhigUK'
    SHELVES_FILENAME = 'goodreads_books.json.gz'

    GROUPING_DICT = {}
    DROPPED = []
    POST_PROCESS = []

    @property
    def DEFAULT_FILENAME(self) -> str:
        return self.REMOTE_FILENAME

    def _download_dataset(self):
        if not (os.path.exists(os.path.join(self.path, self.DEFAULT_FILENAME)) and \
                os.path.exists(os.path.join(self.path, self.SHELVESURL))):
            raise FileNotFoundError(f'Manually download the dataset from {self.DATASETURL} and place it under {self.path}. To prepare ground truth generative factors manually download also the shelves from {self.SHELVESURL} and place it under {self.path}.')

    def _load_dataframe(self) -> pd.DataFrame:
        self.fetch_dataset()
        return pd.read_csv(
            self.file_path,
            dtype={
                self.USER_IX: np.int64,
                self.ITEM_IX: np.int64,
                self.RATING_IX: np.float64,
            },
            sep=',',
            usecols=[self.USER_IX, self.ITEM_IX, self.RATING_IX]
        )

    def _load_shelves(self) -> pd.DataFrame:
        shelves_file = os.path.join(self.path, self.SHELVES_FILENAME)
        if not os.path.exists(shelves_file):
            raise ValueError(f'Manually download the shelves from {self.SHELVESURL} and place it under {self.path}')
        
        books, shelves = [], []
        with gzip.open(shelves_file, 'rb') as f:
            for line in tqdm(f, desc=f'Parsing {self.SHELVES_FILENAME}'):
                d = json.loads(line)
                book_id = d['book_id']
                popular_shelves = d['popular_shelves']
                shelves.extend([str(shelf['name']) for shelf in popular_shelves])
                books.extend([int(book_id)] * len(popular_shelves))
        return pd.DataFrame({self.ITEM_IX: books, 'shelf': shelves})

    def get_factors(self, interactions: pd.DataFrame) -> Tuple[sps.spmatrix, dict, dict]:
        book_interactions = interactions.groupby(self.ITEM_IX, as_index=False).agg({self.USER_IX: 'count'})
        interactions_unique_books = book_interactions[self.ITEM_IX].unique()

        shelves = self._load_shelves()
        shelves_books = shelves.groupby('shelf', as_index=False).agg({self.ITEM_IX: list})
        shelves_books['count'] = shelves_books[self.ITEM_IX].map(len)
        sorted_shelves = shelves_books.sort_values('count', ascending=False)

        rows, cols = [], []
        covered_books = set()

        # Iterate over the sorted shelves and fill in rows and cols for the book-shelf matrix
        for _, row in tqdm(sorted_shelves.iterrows(), desc='Computing factors'):
            shelf_name = self.GROUPING_DICT.get(row['shelf'], row['shelf']).lower()
            if shelf_name not in self.DROPPED:
                shelf_books = set(row[self.ITEM_IX]).intersection(interactions_unique_books)
                if shelf_name not in self.POST_PROCESS:
                    rows.extend(shelf_books)
                    cols.extend([shelf_name] * len(shelf_books))
                covered_books = covered_books.union(shelf_books).intersection(interactions_unique_books)
                books_portion_covered = round(len(covered_books) / len(interactions_unique_books) * 1000) / 1000
                if books_portion_covered > 0.99:
                    break

        
        '''Post processing
        Case 1: If a shelf name is composed by two other shelf names (separated by either `-` or space), we delete it
                and copy the books to the two corresponding shelves.
        '''
        unique_shelves = np.unique(cols)
        for shelf in self.POST_PROCESS:
            shelf_books = set(sorted_shelves.loc[sorted_shelves['shelf'] == shelf, self.ITEM_IX].values[0])
            new_unique_books = shelf_books.intersection(interactions_unique_books)

            if '-' in shelf:
                parts = shelf.split('-')
            elif ' ' in shelf:
                parts = shelf.split(' ')
            else:
                continue
            
            if np.all([part in unique_shelves for part in parts]):
                for part in parts:
                    corr_books = set(sorted_shelves.loc[sorted_shelves['shelf'] == part, self.ITEM_IX].values[0])
                    books = new_unique_books.union(corr_books.intersection(interactions_unique_books))
                    rows.extend(books)
                    cols.extend([part] * len(books))


        # Return the book-shelf matrix
        unique_shelves = np.unique(cols)
        book2row = dict(zip(interactions_unique_books, range(len(interactions_unique_books))))
        shelf2col = dict(zip(unique_shelves, range(len(unique_shelves))))
        mapped_books = list(map(book2row.get, rows))
        mapped_shelves = list(map(shelf2col.get, cols))
        return sps.coo_matrix((np.ones_like(rows, dtype=np.int8), (mapped_books, mapped_shelves)),
                              shape=(len(interactions_unique_books), len(unique_shelves))), book2row, shelf2col
