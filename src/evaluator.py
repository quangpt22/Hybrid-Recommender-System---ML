import numpy as np


def build_ranking_data_loo(rate_train, rate_test, n_items):
    # single positive per user (LOO: exactly one held-out item)
    user_pos = {}
    user_neg = {}
    all_items = set(range(n_items))

    train_dict = {}
    for row in rate_train:
        u_idx = int(row[0] - 1)
        i_idx = int(row[1] - 1)
        train_dict.setdefault(u_idx, set()).add(i_idx)

    for row in rate_test:
        u_idx = int(row[0] - 1)
        i_idx = int(row[1] - 1)
        user_pos[u_idx] = i_idx
        train_items = train_dict.get(u_idx, set())
        candidates = list(all_items - train_items - {i_idx})
        user_neg[u_idx] = np.array(candidates)

    return user_pos, user_neg


def build_ranking_data_multi(rate_train, rate_test, n_items):
    # multiple positives per user (80/20: each user may have several test items).
    user_pos = {}
    user_neg = {}
    all_items = set(range(n_items))

    train_dict = {}
    for row in rate_train:
        u_idx = int(row[0] - 1)
        i_idx = int(row[1] - 1)
        train_dict.setdefault(u_idx, set()).add(i_idx)

    test_dict = {}
    for row in rate_test:
        u_idx = int(row[0] - 1)
        i_idx = int(row[1] - 1)
        test_dict.setdefault(u_idx, set()).add(i_idx)

    for u_idx, pos_items in test_dict.items():
        user_pos[u_idx] = list(pos_items)
        train_items = train_dict.get(u_idx, set())
        candidates = list(all_items - train_items - pos_items)
        user_neg[u_idx] = np.array(candidates)

    return user_pos, user_neg


def tune_alphas(pred_item, pred_user, pred_content, rate_train, rate_val,
                n_ratings, n_items, threshold=30, top_k=10, split_type='loo'):

    alphas = np.arange(0, 1.05, 0.05)
    build_fn = build_ranking_data_loo if split_type == 'loo' else build_ranking_data_multi

    def score_user(u_idx, items, alpha, beta, gamma):
        return (alpha * pred_item[u_idx, items] +
                beta * pred_user[u_idx, items] +
                gamma * pred_content[u_idx, items])

    def compute_ndcg(ranked, pos_items, top_k):
        if split_type == 'loo':
            rank_pos = np.where(ranked == pos_items)[0]
            rank = rank_pos[0] + 1 if len(rank_pos) > 0 else len(ranked) + 1
            return 1.0 / np.log2(rank + 1) if rank <= top_k else 0.0
        else:
            pos_set = set(pos_items)
            top_items = ranked[:top_k]
            dcg = sum(1.0 / np.log2(i + 2) for i, it in enumerate(top_items) if it in pos_set)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(pos_items), top_k)))
            return dcg / idcg if idcg > 0 else 0.0

    def tune_group(filter_type):
        filtered_val = []
        for row in rate_val:
            u_idx = int(row[0] - 1)
            n_rated = n_ratings.get(u_idx, 0)
            if filter_type == 'cold' and n_rated >= threshold:
                continue
            if filter_type == 'warm' and n_rated < threshold:
                continue
            filtered_val.append(row)
        filtered_val = np.array(filtered_val)

        if len(filtered_val) == 0:
            print("  No users found for filter_type={}".format(filter_type))
            return 0.0, 0.0, 0.0

        user_pos, user_neg = build_fn(rate_train, filtered_val, n_items)

        best_alpha, best_beta, best_ndcg = 0.0, 0.0, -1.0

        for alpha in alphas:
            for beta in np.arange(0, 1.05 - alpha, 0.05):
                gamma = 1 - alpha - beta
                ndcgs = []

                for u_idx in user_pos:
                    pos_items = user_pos[u_idx]
                    neg_items = user_neg[u_idx].astype(int)

                    if split_type == 'loo':
                        items_to_predict = np.append(neg_items, pos_items)
                    else:
                        items_to_predict = np.concatenate(
                            [neg_items, np.array(pos_items, dtype=int)])

                    scores = score_user(u_idx, items_to_predict, alpha, beta, gamma)
                    ranked = items_to_predict[np.argsort(scores)[::-1]]

                    ndcgs.append(compute_ndcg(ranked, pos_items, top_k))

                mean_ndcg = np.mean(ndcgs)
                if mean_ndcg > best_ndcg:
                    best_ndcg = mean_ndcg
                    best_alpha = alpha
                    best_beta = beta

        print("  alpha={:.2f}, beta={:.2f}, content={:.2f} | NDCG={:.4f}".format(
            best_alpha, best_beta, 1 - best_alpha - best_beta, best_ndcg))

        return best_alpha, best_beta, best_ndcg

    print("\nTuning for cold users (< {} ratings)...".format(threshold))
    alpha_cold, beta_cold, ndcg_cold = tune_group('cold')

    print("\nTuning for warm users (>= {} ratings)...".format(threshold))
    alpha_warm, beta_warm, ndcg_warm = tune_group('warm')

    print("\nBest parameters:")
    print("  Cold: alpha={:.2f}, beta={:.2f}, content={:.2f} | NDCG={:.4f}".format(
        alpha_cold, beta_cold, 1 - alpha_cold - beta_cold, ndcg_cold))
    print("  Warm: alpha={:.2f}, beta={:.2f}, content={:.2f} | NDCG={:.4f}".format(
        alpha_warm, beta_warm, 1 - alpha_warm - beta_warm, ndcg_warm))

    return (alpha_cold, beta_cold), (alpha_warm, beta_warm)


def evaluate_hybrid_matrix(pred_item, pred_user, pred_content, rate_train, rate_test,
                            n_ratings, n_items, alpha_cold, beta_cold, alpha_warm,
                            beta_warm, threshold=30, top_k=10, split_type='loo'):

    build_fn = build_ranking_data_loo if split_type == 'loo' else build_ranking_data_multi

    print("\nBuilding ranking data for evaluation...")
    user_pos, user_neg = build_fn(rate_train, rate_test, n_items)

    # RMSE
    mse, count = 0, 0
    for row in rate_test:
        u_idx = int(row[0] - 1)
        i_idx = int(row[1] - 1)
        r = row[2]

        n_rated = n_ratings.get(u_idx, 0)
        alpha, beta = (alpha_cold, beta_cold) if n_rated < threshold else (alpha_warm, beta_warm)
        gamma = 1 - alpha - beta

        pred = (alpha * pred_item[u_idx, i_idx] +
                beta * pred_user[u_idx, i_idx] +
                gamma * pred_content[u_idx, i_idx])
        pred = np.clip(pred, 1, 5)
        mse += (pred - r) ** 2
        count += 1
    rmse = np.sqrt(mse / count) if count > 0 else 0

    precisions, recalls, ndcgs, mrrs = [], [], [], []

    for u_idx in user_pos:
        pos_items = user_pos[u_idx]
        neg_items = user_neg[u_idx].astype(int)

        n_rated = n_ratings.get(u_idx, 0)
        alpha, beta = (alpha_cold, beta_cold) if n_rated < threshold else (alpha_warm, beta_warm)
        gamma = 1 - alpha - beta

        if split_type == 'loo':
            items_to_predict = np.append(neg_items, pos_items)
        else:
            items_to_predict = np.concatenate([neg_items, np.array(pos_items, dtype=int)])

        scores = (alpha * pred_item[u_idx, items_to_predict] +
                  beta * pred_user[u_idx, items_to_predict] +
                  gamma * pred_content[u_idx, items_to_predict])

        ranked = items_to_predict[np.argsort(scores)[::-1]]
        top_k_items = ranked[:top_k]

        if split_type == 'loo':
            hit = int(pos_items in top_k_items)
            precisions.append(hit / top_k)
            recalls.append(hit)

            rank_pos = np.where(ranked == pos_items)[0]
            rank = rank_pos[0] + 1 if len(rank_pos) > 0 else len(ranked) + 1
            if rank <= top_k:
                ndcgs.append(1.0 / np.log2(rank + 1))
                mrrs.append(1.0 / rank)
            else:
                ndcgs.append(0.0)
                mrrs.append(0.0)
        else:
            pos_set = set(pos_items)
            hits = [it for it in top_k_items if it in pos_set]
            hit_count = len(hits)

            precisions.append(hit_count / top_k)
            recalls.append(hit_count / len(pos_items))

            ranks = [i + 1 for i, it in enumerate(ranked) if it in pos_set]
            if ranks and ranks[0] <= top_k:
                mrrs.append(1.0 / ranks[0])
            else:
                mrrs.append(0.0)

            dcg = sum(1.0 / np.log2(i + 2) for i, it in enumerate(top_k_items) if it in pos_set)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(pos_items), top_k)))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        'rmse': rmse,
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'ndcg': np.mean(ndcgs),
        'mrr': np.mean(mrrs)
    }