import numpy as np

class HybridRecommender3Way:
    def __init__(self, item_model, user_model, content_model,
                 utility_sparse, utility_norm_item, utility_norm_user,
                 user_means, item_means, item_features,
                 n_ratings, alpha_cold, beta_cold,
                 alpha_warm, beta_warm, threshold=30):

        self.item_model = item_model
        self.user_model = user_model
        self.content_model = content_model
        self.utility_sparse = utility_sparse
        self.utility_norm_item = utility_norm_item
        self.utility_norm_user = utility_norm_user
        self.user_means = user_means
        self.item_means = item_means
        self.item_features = item_features
        self.utility_csc = utility_sparse.tocsc()
        self.utility_norm_user_csc = utility_norm_user.tocsc()
        self.n_ratings = n_ratings
        self.alpha_cold = alpha_cold
        self.beta_cold = beta_cold
        self.alpha_warm = alpha_warm
        self.beta_warm = beta_warm
        self.threshold = threshold

    def get_params(self, u_idx):
        if self.n_ratings.get(u_idx, 0) < self.threshold:
            return self.alpha_cold, self.beta_cold
        return self.alpha_warm, self.beta_warm

    def predict_for_user_items(self, u_idx, item_indices):
        item_indices = np.atleast_1d(item_indices)
        alpha, beta = self.get_params(u_idx)
        gamma = 1 - alpha - beta

        scores_item = self.item_model.predict_for_user_items(
            u_idx, item_indices,
            self.utility_sparse, self.utility_norm_item,
            self.user_means, self.item_means
        )
        scores_user = self.user_model.predict_for_user_items(
            u_idx, item_indices,
            self.utility_sparse, self.utility_norm_user,
            self.user_means, self.item_means,
            utility_csc=self.utility_csc,
            utility_norm_csc=self.utility_norm_user_csc
        )
        scores_content = self.content_model.predict(u_idx, item_indices, self.item_features)

        blended = (alpha * scores_item + beta * scores_user + gamma * scores_content)
        return np.clip(blended, 1, 5)