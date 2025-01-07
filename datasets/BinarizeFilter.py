#!/usr/bin/env python3

from recpack.preprocessing.filters import Filter


class BinarizeFilter(Filter):
    """
    This filter is entirely copied from recpack.preprocessing.filters.MinRating but extends min_rating to be a float value for cases when ratings are recorded in decimal values.
    """
    def __init__(
        self,
        min_rating: float,
        rating_ix: str,
    ):
        self.rating_ix = rating_ix
        self.min_rating = min_rating

    def apply(self, df):
        return df[df[self.rating_ix] >= self.min_rating].copy().drop(columns=self.rating_ix)
    
