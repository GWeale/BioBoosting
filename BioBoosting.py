import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

class BioBoostTree:
    def __init__(self, max_depth=3, min_samples_split=2, min_impurity_decrease=1e-7):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.tree = None

    def fit(self, X, y, residuals, depth=0):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            self.tree = np.mean(y)
            return

        best_feat, best_thresh, best_impurity = self._best_split(X, residuals)
        if best_impurity is None or best_impurity < self.min_impurity_decrease:
            self.tree = np.mean(y)
            return

        idx_left = X[:, best_feat] <= best_thresh
        idx_right = X[:, best_feat] > best_thresh

        self.tree = {'feature': best_feat, 'threshold': best_thresh}
        self.tree['left'] = BioBoostTree(self.max_depth, self.min_samples_split, self.min_impurity_decrease)
        self.tree['left'].fit(X[idx_left], y[idx_left], residuals[idx_left], depth + 1)
        self.tree['right'] = BioBoostTree(self.max_depth, self.min_samples_split, self.min_impurity_decrease)
        self.tree['right'].fit(X[idx_right], y[idx_right], residuals[idx_right], depth + 1)

    def _best_split(self, X, residuals):
        m, n = X.shape
        best_impurity = None
        best_feat = None
        best_thresh = None

        for feat in range(n):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                idx_left = X[:, feat] <= thresh
                idx_right = X[:, feat] > thresh
                if len(residuals[idx_left]) == 0 or len(residuals[idx_right]) == 0:
                    continue

                impurity = self._impurity(residuals, idx_left, idx_right)
                if best_impurity is None or impurity < best_impurity:
                    best_impurity = impurity
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh, best_impurity

    def _impurity(self, residuals, idx_left, idx_right):
        w_left = np.sum(np.abs(residuals[idx_left]))
        w_right = np.sum(np.abs(residuals[idx_right]))
        impurity = w_left * np.var(residuals[idx_left]) + w_right * np.var(residuals[idx_right])
        return impurity

    def predict(self, X):
        if isinstance(self.tree, dict):
            feat = self.tree['feature']
            thresh = self.tree['threshold']
            idx_left = X[:, feat] <= thresh
            idx_right = X[:, feat] > thresh
            y_pred = np.zeros(X.shape[0])
            y_pred[idx_left] = self.tree['left'].predict(X[idx_left])
            y_pred[idx_right] = self.tree['right'].predict(X[idx_right])
            return y_pred
        else:
            return np.full(X.shape[0], self.tree)

class BioBoost:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.loss_history = []
        self.feature_importances_ = defaultdict(float)

    def _pseudo_residuals(self, y, y_pred):
        return -2 * (y - y_pred) / (1 + np.abs(y - y_pred))

    def fit(self, X, y):
        y_pred = np.full(y.shape, np.mean(y))
        for i in range(self.n_estimators):
            residuals = self._pseudo_residuals(y, y_pred)
            tree = BioBoostTree(max_depth=self.max_depth)
            tree.fit(X, y, residuals)
            update = tree.predict(X)
            y_pred += self.learning_rate * update
            self.trees.append(tree)
            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)
            self._update_feature_importances(tree.tree, self.learning_rate)

    def _update_feature_importances(self, tree, weight):
        if isinstance(tree, dict):
            self.feature_importances_[tree['feature']] += weight
            self._update_feature_importances(tree['left'].tree, weight)
            self._update_feature_importances(tree['right'].tree, weight)

    def predict(self, X):
        y_pred = np.full(X.shape[0], np.mean([tree.tree if not isinstance(tree.tree, dict) else 0 for tree in self.trees]))
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss over iterations')
        plt.show()

def load_bioinformatics_data():
    df = pd.read_csv('gene_expression.csv')
    y = df['target'].values
    X = df.drop('target', axis=1).values
    return X, y

def preprocess_data(X):
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_scaled

def feature_engineering(X):
    poly_features = []
    for i in range(X.shape[1]):
        for j in range(i, X.shape[1]):
            poly_features.append(X[:, i] * X[:, j])
    return np.column_stack([X] + poly_features)

def main():
    X, y = load_bioinformatics_data()
    X = preprocess_data(X)
    X = feature_engineering(X)
    model = BioBoost(n_estimators=50, learning_rate=0.05, max_depth=4)
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    print(f'Mean Squared Error: {mse}')
    model.plot_loss()
    plt.bar(range(len(model.feature_importances_)), list(model.feature_importances_.values()))
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.show()

if __name__ == '__main__':
    main()
