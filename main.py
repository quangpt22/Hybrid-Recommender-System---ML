import os
import sys
import argparse
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scipy.sparse import csr_matrix

from src.data_loader import MovieLensDataLoader
from src.matrix_generator import (build_item_features, generate_mem_pred_matrix, generate_content_pred_matrix, load_matrix, build_dense_utility)
from src.evaluator import tune_alphas, evaluate_hybrid_matrix
from src.hybrid_model import HybridRecommender3Way
from Memory_based_Filtering.model import MemoryBasedCF
from Content_based_Filtering.model import ContentBasedRecommender


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_params(params_cold, params_warm, config_path):
    cfg = load_config(config_path)
    cfg['hybrid']['alpha_cold'] = round(float(params_cold[0]), 2)
    cfg['hybrid']['beta_cold'] = round(float(params_cold[1]), 2)
    cfg['hybrid']['alpha_warm'] = round(float(params_warm[0]), 2)
    cfg['hybrid']['beta_warm'] = round(float(params_warm[1]), 2)
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print("Saved parameters to {}".format(config_path))


def load_params(config_path):
    cfg = load_config(config_path)
    h = cfg['hybrid']
    if any(h[k] is None for k in ['alpha_cold', 'beta_cold', 'alpha_warm', 'beta_warm']):
        raise ValueError("Hybrid params not found. Run --mode train first.")
    return (h['alpha_cold'], h['beta_cold']), (h['alpha_warm'], h['beta_warm'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['train', 'predict'])
    parser.add_argument('--split', required=True, choices=['loo', '8020'])
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    config_path = './{}/config.yaml'.format(args.split)
    cfg = load_config(config_path)

    inputs_dir = cfg['paths']['inputs']
    checkpoint_dir = cfg['paths']['checkpoints']
    weights_dir = cfg['paths']['weights']
    threshold = cfg['model']['cold_start_threshold']
    n_neighbors = cfg['model']['n_neighbors']
    top_k = cfg['evaluation']['top_k']

    print("\n3-Way Hybrid Recommender")
    print("Mode: {} | Split: {}".format(args.mode.upper(), args.split.upper()))

    loader = MovieLensDataLoader(data_path=cfg['data']['path'])
    ratings, movies = loader.load_raw_data()

    split_files_exist = (
        os.path.exists(os.path.join(inputs_dir, 'rate_train.npy')) and
        os.path.exists(os.path.join(inputs_dir, 'rate_val.npy')) and
        os.path.exists(os.path.join(inputs_dir, 'rate_test.npy'))
    )

    if args.mode == 'train' and args.force:
        split_files_exist = False

    if args.mode == 'predict':
        if not split_files_exist:
            print("Error: No saved split found for this configuration. Run --mode train first.")
            return
        print("Loading saved split from previous training run...")
        rate_train = np.load(os.path.join(inputs_dir, 'rate_train.npy'))
        rate_val = np.load(os.path.join(inputs_dir, 'rate_val.npy'))
        rate_test = np.load(os.path.join(inputs_dir, 'rate_test.npy'))
    else:  # train mode
        if args.split == 'loo':
            rate_train, rate_val, rate_test = loader.split_loo(ratings)
        else:
            rate_train, rate_val, rate_test = loader.split_8020_per_user(
                ratings, test_size=0.1, val_size=0.1, seed=None)

        # save this split so predict mode uses the exact same one
        os.makedirs(inputs_dir, exist_ok=True)
        np.save(os.path.join(inputs_dir, 'rate_train.npy'), rate_train)
        np.save(os.path.join(inputs_dir, 'rate_val.npy'), rate_val)
        np.save(os.path.join(inputs_dir, 'rate_test.npy'), rate_test)
        print("Saved split to {}".format(inputs_dir))

    rate_train = rate_train.astype(int).astype(float)
    rate_val = rate_val.astype(int).astype(float) if len(rate_val) > 0 else rate_val
    rate_test = rate_test.astype(int).astype(float)

    n_users = loader.n_users
    n_items = loader.n_items
    n_ratings = loader.get_user_rating_counts(rate_train)

    print("Users: {}, Items: {}".format(n_users, n_items))
    print("Train: {}, Val: {}, Test: {}".format(len(rate_train), len(rate_val), len(rate_test)))

    item_features = build_item_features(loader.movie2idx, movies)

    if args.mode == 'train':
        pred_item = generate_mem_pred_matrix(
            rate_train, n_users, n_items, mode='item', K=n_neighbors,
            force_recompute=args.force, inputs_dir=inputs_dir,
            checkpoint_dir=checkpoint_dir
        )
        pred_user = generate_mem_pred_matrix(
            rate_train, n_users, n_items, mode='user', K=n_neighbors,
            force_recompute=args.force, inputs_dir=inputs_dir,
            checkpoint_dir=checkpoint_dir
        )
        pred_content = generate_content_pred_matrix(
            rate_train, item_features, n_users, n_items,
            force_recompute=args.force, inputs_dir=inputs_dir,
            weights_dir=weights_dir
        )

        print("\npred_item shape: {}".format(pred_item.shape))
        print("pred_user shape: {}".format(pred_user.shape))
        print("pred_content shape: {}".format(pred_content.shape))

        params_cold, params_warm = tune_alphas(
            pred_item, pred_user, pred_content,
            rate_train, rate_val, n_ratings, n_items,
            threshold, top_k, split_type=args.split
        )
        save_params(params_cold, params_warm, config_path)
        print("\nTraining done. Run python main.py --mode predict --split {} to evaluate.".format(args.split))

    else:
        params_cold, params_warm = load_params(config_path)
        alpha_cold, beta_cold = params_cold
        alpha_warm, beta_warm = params_warm

        print("\nLoaded params:")
        print("  Cold: alpha={:.2f}, beta={:.2f}, content={:.2f}".format(alpha_cold, beta_cold, 1 - alpha_cold - beta_cold))
        print("  Warm: alpha={:.2f}, beta={:.2f}, content={:.2f}".format(alpha_warm, beta_warm, 1 - alpha_warm - beta_warm))

        pred_item = load_matrix('pred_item.npy', inputs_dir)
        pred_user = load_matrix('pred_user.npy', inputs_dir)
        pred_content = load_matrix('pred_content.npy', inputs_dir)

        if pred_item is None or pred_user is None or pred_content is None:
            print("Error: Missing pred matrices. Run --mode train first.")
            return

        results = evaluate_hybrid_matrix(
            pred_item, pred_user, pred_content,
            rate_train, rate_test, n_ratings, n_items,
            alpha_cold, beta_cold, alpha_warm, beta_warm,
            threshold, top_k, split_type=args.split
        )

        print("\nFinal Results ({} split):".format(args.split.upper()))
        print("  RMSE:         {:.4f}".format(results['rmse']))
        print("  Precision@{}: {:.4f}".format(top_k, results['precision']))
        print("  Recall@{}:    {:.4f}".format(top_k, results['recall']))
        print("  NDCG@{}:      {:.4f}".format(top_k, results['ndcg']))
        print("  MRR@{}:       {:.4f}".format(top_k, results['mrr']))

        # Inference: recommend for a random real user using the saved weights.
        print("\nINFERENCE: TOP RECOMMENDATIONS FOR A RANDOM USER")

        # Rebuild the base models
        Utility, Unorm_item, user_means, item_means = build_dense_utility(rate_train, n_users, n_items, mode='item')
        _, Unorm_user, _, _ = build_dense_utility(rate_train, n_users, n_items, mode='user')

        # Build sparse matrices
        rows = (rate_train[:, 0] - 1).astype(int)
        cols = (rate_train[:, 1] - 1).astype(int)
        utility_sparse = csr_matrix((Utility[rows, cols], (rows, cols)), shape=(n_users, n_items))
        utility_norm_item = csr_matrix((Unorm_item[rows, cols], (rows, cols)), shape=(n_users, n_items))
        utility_norm_user = csr_matrix((Unorm_user[rows, cols], (rows, cols)), shape=(n_users, n_items))

        item_model = MemoryBasedCF(mode='item', n_neighbors=n_neighbors)
        item_model.load_checkpoint(os.path.join(checkpoint_dir, 'item_cf_checkpoint.pkl'))
        user_model = MemoryBasedCF(mode='user', n_neighbors=n_neighbors)
        user_model.load_checkpoint(os.path.join(checkpoint_dir, 'user_cf_checkpoint.pkl'))

        content_model = ContentBasedRecommender(n_users, item_features.shape[1])
        content_model.load_weights(os.path.join(weights_dir, 'cb_weights.npz'))

        hybrid = HybridRecommender3Way(
            item_model, user_model, content_model,
            utility_sparse, utility_norm_item, utility_norm_user,
            user_means, item_means, item_features,
            n_ratings, alpha_cold, beta_cold, alpha_warm, beta_warm, threshold)

        # items this user already rated in train (exclude from recommendations)
        train_by_user = {}
        for row in rate_train:
            u_idx = int(row[0] - 1)
            i_idx = int(row[1] - 1)
            train_by_user.setdefault(u_idx, set()).add(i_idx)

        # pick a random user
        rand_user = int(np.random.choice([u for u in train_by_user if train_by_user[u]]))
        rated = train_by_user[rand_user]
        unrated = np.array([i for i in range(n_items) if i not in rated])

        n_rated = n_ratings.get(rand_user, 0)
        alpha, beta = hybrid.get_params(rand_user)
        gamma = 1 - alpha - beta

        scores = hybrid.predict_for_user_items(rand_user, unrated)

        order = np.argsort(scores)[::-1][:top_k]
        top_items = unrated[order]
        top_scores = scores[order]

        user_type = "COLD" if n_rated < threshold else "WARM"
        print("  User ID        : {}".format(rand_user + 1))
        print("  Profile        : {} ({} train ratings)".format(user_type, n_rated))
        print("  Weights used   : alpha={:.2f}, beta={:.2f}, content={:.2f}".format(
            alpha, beta, gamma))
        print("  Top {} recommendations:".format(top_k))

        title_lookup = movies.set_index('movie_id')['title'].to_dict()
        for rank, (i_idx, score) in enumerate(zip(top_items, top_scores), start=1):
            movie_id = loader.idx2movie[int(i_idx)]
            title = title_lookup.get(movie_id, "ID {}".format(movie_id))
            print("    {:>2}. {:<45} (pred {:.2f}/5.0)".format(rank, title, score))


if __name__ == "__main__":
    main()