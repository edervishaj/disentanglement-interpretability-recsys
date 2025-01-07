#!/usr/bin/env python3

# distutils: language=c++

cimport cython
import numpy as np
import scipy.sparse as sps

from libc.math cimport fmin
from libcpp.unordered_set cimport unordered_set

class RelevanceMetrics:
    METRIC_NAMES = ('ndcg', 'recall', 'mrr', 'coverage')

    def __init__(self, metrics: list, Ks: list, URM: sps.csr_matrix):
        assert np.all(np.in1d(metrics, self.METRIC_NAMES)), f'Invalid metric provided. Allowed metrics are {self.METRIC_NAMES}!'

        self.total_users = 0
        self.total_items = URM.shape[1]
        self.metrics = metrics
        self.Ks = Ks
        self._metric_values = {f'{metric}_{k}': 0.0 for metric in self.METRIC_NAMES for k in self.Ks}
        self._metric_values.update({f'coverage_{k}': set() for k in self.Ks})
        
        item_popularity = URM.sum(axis=0).A1
        top_pop_maxK = np.argsort(-item_popularity)[:max(self.Ks)]
        self.top_pop_items = {k: top_pop_maxK[:k] for k in self.Ks}

    def calculate(self, y_true: sps.csr_matrix, y_pred: np.ndarray) -> None:
        top_k = self.compute_top_k(y_pred)
        return self._calculate(y_true, top_k)

    def compute_top_k(self, y_pred: np.ndarray) -> np.ndarray:
        # :max(self.Ks) elements are the smallest (negated --> largest)
        rel_partitioned = np.argpartition(-y_pred, kth=max(self.Ks)-1, axis=1)[:, :max(self.Ks)]

        # retrieve the values of smallest (negated --> largets)
        partitioned_real = y_pred[np.arange(y_pred.shape[0])[:, None], rel_partitioned]

        # sort in decreasing order the real values
        # first element is the top ranked one!
        argsorted_real = np.argsort(-partitioned_real, axis=1)

        return rel_partitioned[np.arange(rel_partitioned.shape[0])[:, None], argsorted_real]
        # return rel_partitioned

    @cython.cdivision(True)
    @cython.boundscheck(False)
    def _calculate(self, y_true: sps.csr_matrix, y_pred: np.ndarray) -> None:
        assert y_true.shape[0] == y_pred.shape[0], f'Predictions for different number of users ({y_pred.shape[0]}) than ground truth ({y_true.shape[0]})!'

        cdef int k, u, i, relevant, total_relevant, unexpected
        cdef int n_users = y_true.shape[0]
        cdef double ideal_cumulative_gain, ndcg, recall, mrr, serendipity

        cdef int[:] y_true_indptr = y_true.indptr
        cdef int[:] y_true_indices = y_true.indices
        cdef int[:, :] ranked = y_pred.astype(np.int32)
        cdef int[:, :] new_items = np.zeros_like(y_pred, dtype=np.int32)

        cdef double[:] max_cumulative_gain, max_cumulative_gain_sum, cumulative_gain, cumulative_gain_sum, len_new_items

        cdef unordered_set[int] interactions, covered

        self.total_users += n_users
        max_cumulative_gain = 1.0 / np.log2(np.arange(2, max(self.Ks) + 2))
        max_cumulative_gain_sum = np.cumsum(max_cumulative_gain)

        for k in self.Ks:
            new_items_arr = np.in1d(y_pred[:, :k].flatten(), self.top_pop_items[k], invert=True).astype(np.int32).reshape((n_users, k))
            new_items = new_items_arr
            len_new_items = new_items_arr.sum(axis=1).astype(np.float64)

            with nogil:
                
                covered.clear()
                recall = 0.0
                ndcg = 0.0
                mrr = 0.0
                serendipity = 0.0
                cumulative_gain = max_cumulative_gain[:k]
                cumulative_gain_sum = max_cumulative_gain_sum[:k]

                for u in range(n_users):
                    interactions.clear()
                    for i in range(y_true_indptr[u], y_true_indptr[u + 1]):
                        interactions.insert(y_true_indices[i])

                    total_relevant = <int> fmin(k, interactions.size())
                    
                    ideal_cumulative_gain = cumulative_gain_sum[total_relevant - 1]

                    relevant = 0
                    unexpected = 0

                    for i in range(k):
                        covered.insert(ranked[u, i])
                        if interactions.find(ranked[u, i]) != interactions.end():
                            relevant += 1

                            if new_items[u, i]:
                                unexpected += 1

                            if relevant == 1:
                                mrr += 1.0 / (i + 1)

                            ndcg += cumulative_gain[i] / ideal_cumulative_gain

                    recall += relevant / (<double> total_relevant)

                    if len_new_items[u] > 0:
                        serendipity += unexpected / len_new_items[u]
            
            self._metric_values[f'ndcg_{k}'] += ndcg
            self._metric_values[f'recall_{k}'] += recall
            self._metric_values[f'mrr_{k}'] += mrr
            if 'coverage' in self.metrics:
                for i in covered:
                    self._metric_values[f'coverage_{k}'].add(i)

    @property
    def values(self):
        total_users = self.total_users if self.total_users > 0 else 1
        results = {}
        for metric, val in self._metric_values.items():
            if metric.startswith('coverage'):
                results[metric] = len(val) / self.total_items
            else:
                results[metric] = val / total_users
        return results
