# 3-Way Hybrid Movie Recommender System (MovieLens 1M)

A hybrid recommender that combines three classic recommendation signals into a
single, cold-start-aware model:

1. **Item-based collaborative filtering** (item–item cosine similarity)
2. **User-based collaborative filtering** (user–user cosine similarity)
3. **Content-based filtering** (per-user Ridge regression over TF-IDF genre features)

The three signals are blended with a **weighted linear combination** whose
weights are **tuned separately for cold and warm users**, so users with little
history lean on content while users with rich history lean on collaborative
filtering.

---

## Table of Contents

- [How the hybrid works](#how-the-hybrid-works)
- [Repository structure](#repository-structure)
- [Quick start](#quick-start)
- [The pipeline, step by step](#the-pipeline-step-by-step)
- [config.yaml (per split)](#configyaml-per-split)
- [Evaluation protocol & the validation set](#evaluation-protocol--the-validation-set)
- [Why prediction matrices?](#why-prediction-matrices)
- [Notes & design decisions](#notes--design-decisions)

---

## How the hybrid works

For a user *u* and item *i*, the predicted score is a weighted blend:

```
score(u, i) = α · item_CF(u, i) + β · user_CF(u, i) + γ · content(u, i)
where γ = 1 − α − β
```

The weights depend on how many ratings the user has in the training set:

- **Cold users** (`< cold_start_threshold` ratings) → `(α_cold, β_cold, γ_cold)`
- **Warm users** (`≥ cold_start_threshold` ratings) → `(α_warm, β_warm, γ_warm)`

The weights are not set by hand — they are grid-searched on the validation set
to maximize NDCG, separately for each user group.

---

## Repository structure

```
.
├── main.py                       # entry point: train / predict for both splits
│
├── src/
│   ├── data_loader.py            # load + preprocess + split MovieLens 1M
│   ├── matrix_generator.py       # item features + full prediction matrices
│   ├── evaluator.py              # weight tuning + hybrid evaluation
│   └── hybrid_model.py           # HybridRecommender3Way (per-user inference)
│
├── Memory_based_Filtering/
│   └── model.py                  # MemoryBasedCF (item & user CF)
│
├── Content_based_Filtering/
│   └── model.py                  # ContentBasedRecommender
│
├── 8020/                         # artifacts for the 70/10/20 split
│   ├── config.yaml               # paths, hyperparameters, tuned weights
│   ├── inputs/                   # split files + prediction matrices (generated)
│   ├── checkpoints/              # CF similarity matrices (generated)
│   └── weights/                  # content model weights (generated)
│
├── loo/                          # artifacts for the leave-one-out split
│   └── (same layout as 8020/)
│
└── data/                         # MovieLens 1M raw files
```

Generated artifacts (`inputs/pred_*.npy`, `checkpoints/`, `weights/`) are
git-ignored — they are rebuilt by `--mode train`.

---

## Quick start

### 1. Requirements

```bash
pip install numpy pandas scipy scikit-learn pyyaml
```

### 2. Data

Place the MovieLens 1M data files in `data/`:

```
data/
├── ratings.dat
├── movies.dat
└── users.dat
```

(Download from https://grouplens.org/datasets/movielens/1m/ if not present.)

### 3. Train (build models + tune the hybrid weights)

```bash
python main.py --mode train --split 8020      # 70/10/20 random split
python main.py --mode train --split loo       # leave-one-out split
```

Add `--force` to re-split the data and recompute everything from scratch:

```bash
python main.py --mode train --split 8020 --force
```

> **Note:** training is slow and can take a long time to finish. There are two
> reasons:
>
> 1. **Building the full prediction matrices.** For each base model we fill an
>    entire `n_users × n_items` matrix (≈ 6,000 × 3,700 ≈ 22 million cells), in
>    pure-Python loops: for every user, and for every unrated item, we gather the
>    top-K neighbors and compute a weighted average. The user-based pass is the
>    worst — for each unrated item it scans the whole rating column to find who
>    rated it, so the cost scales with users × items × neighbors.
> 2. **The weight grid-search.** Tuning then sweeps every (α, β) combination
>    (~200+ of them) and re-ranks the full candidate list for every validation
>    user — twice (cold users and warm users separately).
>
> The good news: this only runs once per split. The matrices and similarity
> checkpoints are cached afterwards, so `predict` (and later runs without
> `--force`) is fast.

### 4. Predict (evaluate on the test set + show a sample recommendation)

```bash
python main.py --mode predict --split 8020
python main.py --mode predict --split loo
```

`predict` prints the final ranking metrics **and** a top-K recommendation list
for a random real user, produced by the actual hybrid model object.

> You must run `train` before `predict` for a given split — `predict` loads the
> split, the prediction matrices and the tuned weights that `train` produced.

---

## The pipeline, step by step

```
raw ratings.dat / movies.dat
        │
        ▼
1. DATA PREPROCESSING        (src/data_loader.py)
   • load ratings & movies
   • remap movie IDs to a compact 0..N-1 space (movie2idx)
   • split per user into train / validation / test
        │
        ▼
2. ITEM FEATURES             (src/matrix_generator.py: build_item_features)
   • multi-hot encode genres → TF-IDF transform
        │
        ▼
3. BASE MODELS → FULL PREDICTION MATRICES   (src/matrix_generator.py)
   • item-CF   → pred_item.npy      (8020/inputs or loo/inputs)
   • user-CF   → pred_user.npy
   • content   → pred_content.npy
   • similarity checkpoints saved to */checkpoints
   • content weights saved to */weights/cb_weights.npz
        │
        ▼
4. TUNE HYBRID WEIGHTS       (src/evaluator.py: tune_alphas)
   • grid-search (alpha, beta, gamma) on the VALIDATION set
   • separately for cold users and warm users
   • best weights saved into the split's config.yaml
        │
        ▼
5. EVALUATE                  (src/evaluator.py: evaluate_hybrid_matrix)
   • blend the three matrices with the tuned weights
   • rank candidates on the TEST set → RMSE, P@K, R@K, NDCG@K, MRR@K
        │
        ▼
6. INFERENCE                 (main.py + src/hybrid_model.py)
   • rebuild the base models, instantiate HybridRecommender3Way
   • recommend top-K unseen movies for a random user
```

---

## config.yaml (per split)

```yaml
data:
  path: ./data
model:
  cold_start_threshold: 30     # users with < 30 train ratings are "cold"
  n_neighbors: 30              # K for KNN collaborative filtering
evaluation:
  test_size: 0.2               # fraction held out for test (8020 split)
  val_size: 0.1                # fraction held out for validation; train = remainder
  top_k: 20                    # K for ranking metrics (8020 uses 20, loo uses 10)
hybrid:                        # filled in by tuning (alpha=item, beta=user)
  alpha_cold: ...
  beta_cold: ...
  alpha_warm: ...
  beta_warm: ...
```

---

## Evaluation protocol & the validation set

Two splitting strategies are supported:

| Split    | Train | Validation | Test | Positives per user |
| -------- | ----- | ---------- | ---- | ------------------ |
| `8020`   | 70%   | 10%        | 20%  | several            |
| `loo`    | all but last 2 | 1 | 1 | exactly one (leave-one-out) |

**Why a separate validation set?** The hybrid has tunable parameters
(α, β per cold/warm group). Those parameters **must be chosen on data the test
set never sees**, otherwise the reported test metrics would be optimistically
biased. So:

- **train** → fits the three base models
- **validation** → tunes the hybrid weights (this is effectively the hybrid's
  "training" data)
- **test** → final, untouched evaluation

The 20% test partition matches the held-out ranking set used by the single-model
baselines, so the ranking metrics are directly comparable. The 10% validation
set — which the standalone baselines do not use — is reserved exclusively for
tuning the hybrid weights, with no test-set leakage. Training on 70% (rather than
the baselines' 80%) is the cost of carving out that validation set; the hybrid
still outperforms every single model despite slightly less training data.

Ranking metrics are computed by scoring each user's held-out positive item(s)
against all of their unseen items, ranking, and measuring Precision@K,
Recall@K, NDCG@K and MRR@K. RMSE is measured on the observed test ratings.

### Results

On the 70/10/20 split, the **tuned hybrid outperforms all three single models**
on the ranking metrics — confirming that fusing complementary signals helps when
there is enough held-out data to both tune and evaluate the blend.

Under **Leave-One-Out**, the hybrid does **not** beat the strongest single model,
and pure content- or user-CF can match or exceed the tuned blend. This is a
property of the protocol, not of the models: LOO leaves exactly **one** item per
user for validation, which is too little signal to tune a three-way blend — the
weights that rank that single validation item best do not generalize to the
single (different) test item, so the blend overfits the sparse validation set. 
The hybrid's advantage therefore appears precisely when the
evaluation provides enough per-user signal (the multi-positive 70/10/20 split),
which is absent under single-positive LOO.

---

## Why prediction matrices?

During tuning, the grid search scores every candidate item for many users, many
times (once per weight combination). If we did this by calling each base model's
per-user predict function on the fly, the cost would explode:

- **The expensive work would be repeated for every (α, β) combination.** The
  per-user predictor recomputes a prediction from scratch each call — gathering
  the user's rated items (or an item's raters), slicing the similarity matrix,
  selecting the top-K neighbors, and computing the weighted average. None of
  that depends on α or β, yet it would be redone on every one of the ~200+ grid
  steps, for every user, for every candidate item. That is the same heavy
  computation run hundreds of times over.
- **It runs in Python per (user, item), with no vectorization.** Each call does
  its own argsort and dot product on small arrays, so the per-call overhead is
  paid millions of times. The user-based predictor is especially costly because
  it scans a rating column to find an item's raters on every single call.

Instead we precompute the full `n_users × n_items` prediction matrix for each
base model **once** (this is the slow part of training). After that, tuning and
evaluation are just fast NumPy array indexing and a weighted sum — the heavy CF
work is never repeated, no matter how many weight combinations are tried.

The matrices are numerically identical to calling the per-user predictors — this
was verified to machine precision. Inference (step 6 of the pipeline) uses the
real `HybridRecommender3Way` object to demonstrate the proper "call the model"
pathway, since at inference time we only score one user, so the per-user path is
cheap and there is no repetition to amortize.

---

## Notes & design decisions

- **Non-deterministic split.** `split_8020_per_user` uses `seed=None`, so each
  `--mode train --force` draws a fresh random split. Metrics therefore vary
  slightly between runs — this is intentional and reflects genuine variance. To
  make results reproducible, pass a fixed seed in
  `src/data_loader.py: split_8020_per_user`.
- **Fast eval vs. real inference.** Bulk evaluation uses precomputed prediction
  matrices for speed; the per-user inference demo uses the real
  `HybridRecommender3Way` model object. Both paths produce the same scores.
- **Sparse-matrix alignment.** The collaborative-filtering predictors rely on the
  rating matrix and the normalized rating matrix sharing the **same** sparsity
  pattern. Normalized matrices are therefore built from COO triplets (which keep
  explicit zeros) rather than from a dense array (which silently drops them).
- **Generated artifacts are not committed.** Run `--mode train` to rebuild the
  prediction matrices, similarity checkpoints and content weights for each split.
