#!/usr/bin/env python3

from recpack.preprocessing.filters import Filter

class CoreFilter(Filter):
    def __init__(self, user_ix, item_ix, minUsersPerItem=1, minItemsPerUser=1) -> None:
        self.user_ix = user_ix
        self.item_ix = item_ix
        self.minUsersPerItem = minUsersPerItem
        self.minItemsPerUser = minItemsPerUser

    def apply(self, df):
        user_ratings = df.groupby(self.user_ix, as_index=False).agg({self.item_ix: 'count'})
        item_ratings = df.groupby(self.item_ix, as_index=False).agg({self.user_ix: 'count'})
        while min(user_ratings[self.item_ix]) < self.minItemsPerUser or \
            min(item_ratings[self.user_ix]) < self.minUsersPerItem:
            keep_users = user_ratings.loc[user_ratings[self.item_ix] >= self.minItemsPerUser, self.user_ix]
            df = df.loc[df[self.user_ix].isin(keep_users)]

            keep_items = item_ratings.loc[item_ratings[self.user_ix] >= self.minUsersPerItem, self.item_ix]
            df = df.loc[df[self.item_ix].isin(keep_items)]

            user_ratings = df.groupby(self.user_ix, as_index=False).agg({self.item_ix: 'count'})
            item_ratings = df.groupby(self.item_ix, as_index=False).agg({self.user_ix: 'count'})

        return df.copy()
