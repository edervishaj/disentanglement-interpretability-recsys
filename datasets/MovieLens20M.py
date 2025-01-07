#!/usr/bin/env python3

import numpy as np
import pandas as pd
from recpack.datasets.movielens import MovieLensDataset


class MovieLens20M(MovieLensDataset):
    REMOTE_FILENAME = "ratings.csv"
    REMOTE_ZIPNAME = "ml-20m"

    def _load_dataframe(self) -> pd.DataFrame:
        self.fetch_dataset()
        return pd.read_table(
            self.file_path,
            dtype={
                self.USER_IX: np.int64,
                self.ITEM_IX: np.int64,
                self.RATING_IX: np.float64,
                self.TIMESTAMP_IX: np.int64,
            },
            header=0,
            sep=',',
            names=[self.USER_IX, self.ITEM_IX, self.RATING_IX, self.TIMESTAMP_IX],
        )
