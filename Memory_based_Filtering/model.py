import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class MemoryBasedCF:
    def __init__(self, mode='item', n_neighbors=30):
        self.mode = mode
        self.n_neighbors = n_neighbors
        self.similarity_matrix = None
        self.global_mean = 3.5

    def compute_similarity(self, utility_norm_sparse):
        print("-> Computing cosine similarity for mode: {}...".format(self.mode.upper()))
        if self.mode == 'item':
            sim = cosine_similarity(utility_norm_sparse.T)
        else:
            sim = cosine_similarity(utility_norm_sparse)

        np.fill_diagonal(sim, -np.inf)
        self.similarity_matrix = sim.astype(np.float32)
        print("-> Similarity computation complete.")
        return self.similarity_matrix

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'mode': self.mode,
            'n_neighbors': self.n_neighbors,
            'similarity_matrix': self.similarity_matrix
        }
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print("-> Saved similarity checkpoint to: {}".format(path))

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            return False
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        self.mode = checkpoint['mode']
        self.n_neighbors = checkpoint['n_neighbors']
        self.similarity_matrix = checkpoint['similarity_matrix']
        print("-> Loaded similarity checkpoint from: {}".format(path))
        return True

    def predict_for_user_items(self, u_idx, item_indices, utility_sparse,
                                utility_norm_sparse, user_means, item_means,
                                utility_csc=None, utility_norm_csc=None):
        item_indices = np.atleast_1d(item_indices)
        preds = np.zeros(len(item_indices), dtype=np.float32)

        if self.mode == 'item':
            start_idx = utility_sparse.indptr[u_idx]
            end_idx = utility_sparse.indptr[u_idx + 1]
            rated_items = utility_sparse.indices[start_idx:end_idx]
            rated_vals_norm = utility_norm_sparse.data[start_idx:end_idx]

            if len(rated_items) == 0:
                fallback = item_means[item_indices].copy()
                fallback[fallback == 0] = self.global_mean
                return fallback

            sim_slice = self.similarity_matrix[item_indices[:, None], rated_items]
            k = min(self.n_neighbors, len(rated_items))
            top_k_indices = np.argsort(sim_slice, axis=1)[:, -k:]

            for idx in range(len(item_indices)):
                k_neighbors = top_k_indices[idx]
                top_sim = sim_slice[idx, k_neighbors]
                denom = np.sum(np.abs(top_sim)) + 1e-8
                num = np.dot(top_sim, rated_vals_norm[k_neighbors])
                preds[idx] = (num / denom) + item_means[item_indices[idx]]

        else:
            sim_vec = self.similarity_matrix[u_idx]

            if utility_csc is None:
                utility_csc = utility_sparse.tocsc()
            if utility_norm_csc is None:
                utility_norm_csc = utility_norm_sparse.tocsc()

            for idx, j in enumerate(item_indices):
                start_idx = utility_csc.indptr[j]
                end_idx = utility_csc.indptr[j + 1]
                users_rated = utility_csc.indices[start_idx:end_idx]
                users_vals_norm = utility_norm_csc.data[start_idx:end_idx]

                if len(users_rated) == 0:
                    preds[idx] = item_means[item_indices[idx]]
                    continue

                sim_slice = sim_vec[users_rated]
                k = min(self.n_neighbors, len(users_rated))
                top_k_idx = np.argsort(sim_slice)[-k:]
                top_sim = sim_slice[top_k_idx]
                denom = np.sum(np.abs(top_sim)) + 1e-8
                num = np.dot(top_sim, users_vals_norm[top_k_idx])
                preds[idx] = (num / denom) + user_means[u_idx]

        return np.clip(preds, 1, 5)

    def generate_full_pred_matrix(self, Utility, Utility_normalized, user_mean, item_mean, K=30):
        n_users, n_items = Utility.shape
        pred = np.zeros((n_users, n_items), dtype=np.float32)

        if self.mode == 'item':
            for u in range(n_users):
                if u % 500 == 0:
                    print("  item-CF: processing user {}/{}...".format(u, n_users))
                rated_items = np.where(Utility[u] > 0)[0]
                unrated_items = np.where(Utility[u] == 0)[0]

                if len(rated_items) == 0:
                    fallback = item_mean[unrated_items].copy()
                    fallback[fallback == 0] = self.global_mean
                    pred[u, unrated_items] = fallback
                    continue

                for j in unrated_items:
                    sim_vec = self.similarity_matrix[j, rated_items]

                    top_k_idx = np.argsort(sim_vec)[-K:]
                    top_items = rated_items[top_k_idx]
                    top_sim = sim_vec[top_k_idx]

                    if len(top_sim) == 0:
                        continue

                    num = np.dot(top_sim, Utility_normalized[u, top_items])
                    denom = np.sum(np.abs(top_sim)) + 1e-8
                    pred[u, j] = (num / denom) + item_mean[j]

        else:  # user-based
            for u in range(n_users):
                if u % 500 == 0:
                    print("  user-CF: processing user {}/{}...".format(u, n_users))
                unrated_items = np.where(Utility[u] == 0)[0]
                for j in unrated_items:
                    users_rated = np.where(Utility[:, j] > 0)[0]

                    if len(users_rated) == 0:
                        pred[u, j] = item_mean[j]
                        continue

                    sim_vec = self.similarity_matrix[u, users_rated]
                    top_k_idx = np.argsort(sim_vec)[-K:]
                    top_users = users_rated[top_k_idx]
                    top_sim = sim_vec[top_k_idx]

                    num = np.dot(top_sim, Utility_normalized[top_users, j])
                    denom = np.sum(np.abs(top_sim)) + 1e-8
                    pred[u, j] = (num / denom) + user_mean[u]

        return np.clip(pred, 1, 5)