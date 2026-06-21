import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from Memory_based_Filtering.model import MemoryBasedCF
from Content_based_Filtering.model import ContentBasedRecommender


def save_matrix(matrix, filename, folder):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    np.save(path, matrix)
    print("Saved matrix to {}".format(path))
    return path


def load_matrix(filename, folder):
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        return None
    matrix = np.load(path)
    print("Loaded matrix from {}".format(path))
    return matrix


def build_item_features(movie2idx, movies_df):
    movies_df = movies_df[movies_df['movie_id'].isin(movie2idx.keys())].copy()
    movies_df['m_idx'] = movies_df['movie_id'].map(movie2idx)
    movies_df = movies_df.sort_values('m_idx')
    movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split('|'))

    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movies_df['genres'])

    transformer = TfidfTransformer(smooth_idf=True, norm='l2')
    item_features = transformer.fit_transform(genre_matrix).toarray()
    return item_features


def build_dense_utility(rate_train, n_users, n_items, mode='item'):
    Utility = np.zeros((n_users, n_items), dtype=np.float32)
    for row in rate_train:
        u = int(row[0]); i = int(row[1])
        Utility[u - 1, i - 1] = row[2]

    user_means = np.zeros(n_users, dtype=np.float32)
    item_means = np.zeros(n_items, dtype=np.float32)

    for i in range(n_users):
        rated = np.where(Utility[i] > 0)[0]
        if len(rated) > 0:
            user_means[i] = Utility[i, rated].mean()

    for j in range(n_items):
        rated = np.where(Utility[:, j] > 0)[0]
        if len(rated) > 0:
            item_means[j] = Utility[rated, j].mean()

    Utility_normalized = Utility.copy()
    if mode == 'user':
        for i in range(n_users):
            rated = np.where(Utility[i] > 0)[0]
            Utility_normalized[i, rated] -= user_means[i]
    else:
        for j in range(n_items):
            rated = np.where(Utility[:, j] > 0)[0]
            Utility_normalized[rated, j] -= item_means[j]

    return Utility, Utility_normalized, user_means, item_means


def generate_mem_pred_matrix(rate_train, n_users, n_items, mode='item', K=30,
                              force_recompute=False, inputs_dir='./inputs',
                              checkpoint_dir='./checkpoints'):
    pred_filename = 'pred_{}.npy'.format(mode)
    checkpoint_path = os.path.join(checkpoint_dir, '{}_cf_checkpoint.pkl'.format(mode))

    if not force_recompute:
        pred = load_matrix(pred_filename, inputs_dir)
        if pred is not None:
            return pred

    print("Building dense utility matrix for {} mode...".format(mode))
    Utility, Utility_norm, user_means, item_means = build_dense_utility(
        rate_train, n_users, n_items, mode=mode)

    model = MemoryBasedCF(mode=mode, n_neighbors=K)

    if force_recompute or not model.load_checkpoint(checkpoint_path):
        print("Computing {}-based similarity matrix...".format(mode))
        from scipy.sparse import csr_matrix
        model.compute_similarity(csr_matrix(Utility_norm))
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_checkpoint(checkpoint_path)

    print("Generating {}-based full pred matrix...".format(mode))
    pred = model.generate_full_pred_matrix(Utility, Utility_norm, user_means, item_means, K)

    save_matrix(pred, pred_filename, inputs_dir)
    return pred


def generate_content_pred_matrix(rate_train, item_features, n_users, n_items,
                                  force_recompute=False, inputs_dir='./inputs',
                                  weights_dir='./weights'):
    pred_filename = 'pred_content.npy'
    weights_path = os.path.join(weights_dir, 'cb_weights.npz')

    if not force_recompute:
        pred = load_matrix(pred_filename, inputs_dir)
        if pred is not None and pred.shape == (n_users, n_items):
            return pred

    model = ContentBasedRecommender(n_users, item_features.shape[1], alpha=1.0)

    if os.path.exists(weights_path) and not force_recompute:
        print("Loading content-based weights...")
        model.load_weights(weights_path)
    else:
        print("Training content-based model...")
        train_df = pd.DataFrame(rate_train, columns=['user_id', 'movie_id', 'rating'])
        train_df['u_idx'] = (train_df['user_id'] - 1).astype(int)
        train_df['m_idx'] = (train_df['movie_id'] - 1).astype(int)
        model.fit(train_df, item_features)
        os.makedirs(weights_dir, exist_ok=True)
        model.save_weights(weights_path)

    print("Generating content-based full pred matrix...")
    pred = model.generate_full_pred_matrix(item_features)
    save_matrix(pred, pred_filename, inputs_dir)
    return pred