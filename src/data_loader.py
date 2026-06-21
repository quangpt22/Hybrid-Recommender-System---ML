import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class MovieLensDataLoader:
    def __init__(self, data_path='./data'):
        self.data_path = data_path
        self.n_users = 0
        self.n_items = 0
        self.movie2idx = {}
        self.idx2movie = {}

    def load_raw_data(self):
        rating_path = os.path.join(self.data_path, 'ratings.dat')
        movie_path = os.path.join(self.data_path, 'movies.dat')

        ratings = pd.read_csv(
            rating_path, sep='::', engine='python',
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            encoding='latin-1'
        )
        movies = pd.read_csv(
            movie_path, sep='::', engine='python',
            names=['movie_id', 'title', 'genres'],
            encoding='latin-1'
        )

        rated_movie_ids = sorted(ratings['movie_id'].unique())
        self.movie2idx = {m: i for i, m in enumerate(rated_movie_ids)}
        self.idx2movie = {i: m for m, i in self.movie2idx.items()}
        self.n_users = ratings['user_id'].max()
        self.n_items = len(rated_movie_ids)

        return ratings, movies

    def split_loo(self, ratings):
        ratings = ratings.sort_values(['user_id', 'timestamp'])
        train_list, val_list, test_list = [], [], []

        for user_id, group in ratings.groupby('user_id'):
            if len(group) < 3:
                train_list.append(group)
                continue
            test_indices = group.index[-1:]
            val_indices = group.index[-2:-1]
            train_indices = group.index[:-2]
            train_list.append(group.loc[train_indices])
            val_list.append(group.loc[val_indices])
            test_list.append(group.loc[test_indices])

        return (self._to_numpy(train_list),
                self._to_numpy(val_list),
                self._to_numpy(test_list))

    def split_8020_per_user(self, ratings, test_size=0.1, val_size=0.1, seed=None):
        rng = np.random.default_rng(seed)
        train_list, val_list, test_list = [], [], []

        for user_id, group in ratings.groupby('user_id'):
            n = len(group)
            shuffled = group.sample(frac=1, random_state=None if seed is None
                                     else int(rng.integers(0, 1e6)))

            n_test = max(1, int(n * test_size)) if n >= 3 else 0
            n_val = max(1, int(n * val_size)) if n - n_test >= 2 else 0

            test_part = shuffled.iloc[:n_test]
            val_part = shuffled.iloc[n_test:n_test + n_val]
            train_part = shuffled.iloc[n_test + n_val:]

            if len(train_part) == 0:
                train_part = shuffled
                val_part = shuffled.iloc[0:0]
                test_part = shuffled.iloc[0:0]

            train_list.append(train_part)
            if len(val_part) > 0:
                val_list.append(val_part)
            if len(test_part) > 0:
                test_list.append(test_part)

        return (self._to_numpy(train_list), self._to_numpy(val_list), self._to_numpy(test_list))

    def _to_numpy(self, df_list):
        if not df_list:
            return np.array([])
        df = pd.concat(df_list)
        arr = df[['user_id', 'movie_id', 'rating']].to_numpy().copy().astype(float)
        arr[:, 1] = np.array([self.movie2idx[m] + 1 for m in arr[:, 1]])
        return arr.astype(int).astype(float)

    def build_utility_matrix(self, rate_train, mode='item'):
        rows = rate_train[:, 0].astype(int) - 1
        cols = rate_train[:, 1].astype(int) - 1
        data = rate_train[:, 2].astype(np.float32)

        utility_sparse = csr_matrix(
            (data, (rows, cols)), shape=(self.n_users, self.n_items))

        user_sums = np.array(utility_sparse.sum(axis=1)).flatten()
        user_counts = np.array((utility_sparse > 0).sum(axis=1)).flatten()
        user_means = np.zeros(self.n_users, dtype=np.float32)
        nz_users = user_counts > 0
        user_means[nz_users] = user_sums[nz_users] / user_counts[nz_users]

        item_sums = np.array(utility_sparse.sum(axis=0)).flatten()
        item_counts = np.array((utility_sparse > 0).sum(axis=0)).flatten()
        item_means = np.zeros(self.n_items, dtype=np.float32)
        nz_items = item_counts > 0
        item_means[nz_items] = item_sums[nz_items] / item_counts[nz_items]

        if mode == 'user':
            normalized_data = data - user_means[rows]
        else:
            normalized_data = data - item_means[cols]

        utility_norm_sparse = csr_matrix((normalized_data, (rows, cols)), shape=(self.n_users, self.n_items))

        global_mean = float(data.mean()) if len(data) > 0 else 3.5

        return utility_sparse, utility_norm_sparse, user_means, item_means, global_mean

    def get_user_rating_counts(self, rate_train):
        counts = {}
        for row in rate_train:
            u_idx = int(row[0] - 1)
            counts[u_idx] = counts.get(u_idx, 0) + 1
        return counts