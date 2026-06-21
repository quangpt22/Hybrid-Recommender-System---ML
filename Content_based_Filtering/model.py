import numpy as np
from sklearn.linear_model import Ridge


class ContentBasedRecommender:
    def __init__(self, n_users, d_features, alpha=1.0):
        self.W = np.zeros((d_features, n_users))
        self.b = np.zeros((1, n_users))
        self.alpha = alpha
        self.global_mean = 0.0

    def fit(self, train_df, item_features):
        self.global_mean = train_df['rating'].mean()
        grouped = train_df.groupby('u_idx')

        for u_idx, group in grouped:
            m_indices = group['m_idx'].values
            scores = group['rating'].values
            X_u = item_features[m_indices, :]

            model = Ridge(alpha=self.alpha, fit_intercept=True)
            model.fit(X_u, scores)

            self.W[:, u_idx] = model.coef_
            self.b[0, u_idx] = model.intercept_

    def predict(self, u_idx, m_indices, item_features):
        X_u = item_features[m_indices, :]
        return X_u.dot(self.W[:, u_idx]) + self.b[0, u_idx]

    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b, global_mean=np.array([self.global_mean]))
        print("Saved content-based weights to: {}".format(filepath))

    def load_weights(self, filepath):
        data = np.load(filepath)
        self.W = data['W']
        self.b = data['b']
        self.global_mean = float(data['global_mean'][0])
        print("Loaded content-based weights from: {}".format(filepath))

    def generate_full_pred_matrix(self, item_features):
        n_users = self.W.shape[1]
        n_items = item_features.shape[0]
        pred = np.zeros((n_users, n_items), dtype=np.float32)
        for u in range(n_users):
            pred[u, :] = self.predict(u, np.arange(n_items), item_features)
        return np.clip(pred, 1, 5)