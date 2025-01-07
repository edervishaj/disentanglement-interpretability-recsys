#!/usr/bin/env python3

"""
Retrieves ground truth tag clusters for users as in:
"Disentangling Preference Representations for Recommendation Critiquing with 𝛽-VAE"
https://dl.acm.org/doi/pdf/10.1145/3459637.3482425
"""

import os
import zipfile

import numpy as np
import pandas as pd
import scipy.sparse as sps
from recpack.datasets import MovieLens1M
from recpack.matrix import InteractionMatrix
from sklearn.cluster import KMeans

from datasets.AmazonCD import AmazonCD
from datasets.GoodReads import GoodReads
from datasets.MovieLens20M import MovieLens20M
from datasets.Yelp import Yelp

N_CLUSTERS = 20


def _topK_tags(dataset, genome_tags_path, genome_scores_path, M, N, movie_ids, seed):
    tags_df = pd.read_table(genome_tags_path, sep=',')
    scores_df = pd.read_table(genome_scores_path, sep=',')

    scores_tags_df = pd.merge(scores_df, tags_df, on='tagId', left_index=False, right_index=False)
    scores_tags_df.dropna(axis=0, inplace=True)

    # keep only tags on movies for which we have interactions
    scores_tags_df = scores_tags_df[scores_tags_df[dataset.ITEM_IX].isin(movie_ids)]
    
    # select 100 tags with highest mean relevance score across all movies
    mean_tag_scores = scores_tags_df.groupby(by='tagId', as_index=False).agg({'relevance': 'mean'})
    top_tags = mean_tag_scores.sort_values(by='relevance', ascending=False).iloc[:100, 0]
    top_tags_scores = scores_tags_df[(scores_tags_df['tagId'].isin(top_tags)) & (scores_tags_df[dataset.ITEM_IX].isin(movie_ids))]

    # create movie-tag matrix
    uniqueMovieId = np.unique(top_tags_scores['movieId'].values)
    uniqueTagId = np.unique(top_tags_scores['tagId'].values)
    top_tags_scores['newMovieId'] = top_tags_scores['movieId'].map(dict(zip(uniqueMovieId, range(len(uniqueMovieId)))))
    top_tags_scores['newTagId'] = top_tags_scores['tagId'].map(dict(zip(uniqueTagId, range(len(uniqueTagId)))))
    data = top_tags_scores['relevance']
    rows = top_tags_scores['newMovieId']
    cols = top_tags_scores['newTagId']
    movie_tag_mat = sps.coo_matrix((data, (rows, cols)), shape=(len(uniqueMovieId), len(uniqueTagId)),
                                   dtype=np.float32).tocsr()
    
    # cluster tags
    kmeans = KMeans(n_clusters=N, random_state=seed)
    predicted_clusters = kmeans.fit_predict(movie_tag_mat.T.tocsr())
    top_tags_scores['cluster'] = predicted_clusters[top_tags_scores['newTagId']]

    # map each movie to the clusters
    gr_movie_cluster = top_tags_scores.groupby(by=['movieId', 'cluster']).agg({'relevance': 'mean'})
    return top_tags_scores, gr_movie_cluster.index[gr_movie_cluster['relevance'] > M].to_frame(index=False)


def ML1M_tags(d: MovieLens1M, im: InteractionMatrix, sets: list, seed: int):
    M = 0.4

    ml20m_path = d.path.split(os.sep)
    ml20m_path[-1] = 'ML20M'
    ml20m_path = str(os.sep).join(ml20m_path)

    ds = MovieLens20M(path=ml20m_path, use_default_filters=False)
    ds.fetch_dataset()

    with zipfile.ZipFile(os.path.join(ds.path, f"{ds.REMOTE_ZIPNAME}.zip"), "r") as zip_ref:
        zip_ref.extract(f"{ds.REMOTE_ZIPNAME}/genome-tags.csv", ds.path)
        zip_ref.extract(f"{ds.REMOTE_ZIPNAME}/genome-scores.csv", ds.path)
    tags_path = os.path.join(d.path, f"{ds.REMOTE_ZIPNAME}_genome-tags.csv")
    scores_path = os.path.join(d.path, f"{ds.REMOTE_ZIPNAME}_genome-scores.csv")
    os.rename(os.path.join(ds.path, f"{ds.REMOTE_ZIPNAME}/genome-tags.csv"), tags_path)
    os.rename(os.path.join(ds.path, f"{ds.REMOTE_ZIPNAME}/genome-scores.csv"), scores_path)

    # create mapper from d.ITEM_IX to X_train.ITEM_IX
    mapping_cols = [d.ITEM_IX, im.ITEM_IX]
    all_item_mappings = im._df[mapping_cols].drop_duplicates()
    real_to_idx_mapper = dict(zip(all_item_mappings.values[:, 0], all_item_mappings.values[:, 1]))
    
    # create movie-cluster dataframe
    top_tags_scores, movie_cluster_df = _topK_tags(d, tags_path, scores_path, M, N_CLUSTERS,
                                                   im._df[d.ITEM_IX].unique(), seed)
    
    # Save top_tags_scores
    top_tags_scores[im.ITEM_IX] = top_tags_scores[d.ITEM_IX].map(real_to_idx_mapper).astype(np.int32)
    top_tags_scores.dropna(subset=[im.ITEM_IX], inplace=True)
    top_tags_scores.to_csv(os.path.join(d.path, 'top_tags_scores.csv'), index=False)

    # rename movies to match X_train and X_test movie IDs
    movie_cluster_df[im.ITEM_IX] = movie_cluster_df[d.ITEM_IX].map(real_to_idx_mapper)
    movie_cluster_df.dropna(subset=[im.ITEM_IX], inplace=True)
    
    # create movie-cluster matrix
    rows = movie_cluster_df[im.ITEM_IX]
    cols = movie_cluster_df['cluster']
    data = np.ones_like(cols)
    movie_cls_mat = sps.coo_matrix((data, (rows, cols)), (im.num_active_items, N_CLUSTERS), dtype=np.int8).tocsr()

    # compute user-cluster matrices
    user_clusters = [d.values.dot(movie_cls_mat) for d in sets]
    nrm_user_clusters = [uc.multiply(1.0 / sets[i].values.sum(axis=1).A1.reshape(-1, 1)).tocsr() for i, uc in enumerate(user_clusters)]

    # assign only clusters whose movies comprise 50% of user interactions
    return [uc.toarray() >= 0.5 for uc in nrm_user_clusters]


def GoodReads_shelves(dataset: GoodReads, im: InteractionMatrix, sets: list, seed: int):
    interactions_path = os.path.join(dataset.path, dataset.__class__.__name__)
    book_shelf_mat, _, _ = dataset.get_factors(InteractionMatrix.load(interactions_path)._df)
    
    # compute user-shelves matrices
    user_shelves = [d.values.dot(book_shelf_mat) for d in sets]
    nrm_user_shelves = [us.multiply(1.0 / sets[i].values.sum(axis=1).A1.reshape(-1, 1)).tocsr() for i, us in enumerate(user_shelves)]

    # assign only shelves whose books comprise 50% of user interactions
    return [us.toarray() >= 0.5 for us in nrm_user_shelves]


def yelp_categories(dataset: Yelp, im: InteractionMatrix, sets: list, seed: int):
    interactions_path = os.path.join(dataset.path, dataset.__class__.__name__)
    bus2cat_df, bus2cat_mat, _, _ = dataset.get_factors(InteractionMatrix.load(interactions_path)._df)

    # clustering
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=seed)
    predicted_clusters = kmeans.fit_predict(bus2cat_mat.T.tocsr())

    # assign clusters to businesses and build business x cluster sparse matrix
    rows = bus2cat_df[dataset.ITEM_IX]
    cols = predicted_clusters[bus2cat_df['category']]
    data = np.ones_like(rows)
    bus2cluster_mat = sps.coo_matrix((data, (rows, cols)), shape=(max(rows) + 1, max(cols) + 1), dtype=np.int8)
    
    # compute user-cluster matrices
    user_cls = [d.values.dot(bus2cluster_mat) for d in sets]
    nrm_user_cls = [us.multiply(1.0 / sets[i].values.sum(axis=1).A1.reshape(-1, 1)).tocsr() for i, us in enumerate(user_cls)]

    # assign only clusters whose businesses comprise 50% of user interactions
    return [us.toarray() >= 0.5 for us in nrm_user_cls]


def amazon_cd_categories(dataset: AmazonCD, im: InteractionMatrix, sets: list, seed: int):
    cd2cat_df, cd2cat_mat, item2row, cat2col = dataset.get_factors(im._df)

    cd2cat = cd2cat_df.copy()
    cd2cat[im.ITEM_IX] = cd2cat[dataset.ITEM_IX]
    cd2cat[dataset.ITEM_IX] = cd2cat[dataset.ITEM_IX].map(dict(zip(list(item2row.values()), list(item2row.keys()))))
    cd2cat['catid'] = cd2cat['category']
    cd2cat['category'] = cd2cat['category'].map(dict(zip(list(cat2col.values()), list(cat2col.keys()))))
    cd2cat.to_csv(os.path.join(dataset.path, 'cd2cat.csv'), index=False)

    # clustering
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=seed)
    predicted_clusters = kmeans.fit_predict(cd2cat_mat.T.tocsr())

    # assign clusters to CDs and build CD x cluster sparse matrix
    rows = cd2cat_df[dataset.ITEM_IX]
    cols = predicted_clusters[cd2cat_df['category']]
    data = np.ones_like(rows)
    cd2cluster_mat = sps.coo_matrix((data, (rows, cols)), shape=(max(rows) + 1, max(cols) + 1), dtype=np.int8)

    # compute user-cluster matrices
    user_cls = [d.values.dot(cd2cluster_mat) for d in sets]
    nrm_user_cls = [us.multiply(1.0 / sets[i].values.sum(axis=1).A1.reshape(-1, 1)).tocsr() for i, us in enumerate(user_cls)]

    # assign only clusters whose businesses comprise 50% of user interactions
    return [us.toarray() >= 0.5 for us in nrm_user_cls]
