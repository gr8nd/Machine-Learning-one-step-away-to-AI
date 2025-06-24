# Full-featured XGBoost implementation with softmax multiclass, regression, early stopping, AUC, and shrinkage
import numpy as np
from sklearn.metrics import roc_auc_score


class XGBoost:
    def __init__(self, num_classes=1, n_estimators=10, max_depth=3, learning_rate=0.1,
                 reg_lambda=1.0, gamma=0, colsample_bytree=1.0,
                 objective='reg:squarederror'):
        self.num_classes = num_classes
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.colsample_bytree = colsample_bytree
        self.objective = objective
        self.models = []
        self.feature_subsets = []
        self.best_iteration = None

    def _softmax(self, logits):
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def _one_hot(self, y, K):
        onehot = np.zeros((y.size, K))
        onehot[np.arange(y.size), y] = 1
        return onehot

    def _get_grad_hess(self, y_true, y_pred):
        if self.objective == 'reg:squarederror':
            grad = y_pred - y_true
            hess = np.ones_like(y_true)
            return grad, hess
        elif self.objective == 'binary:logistic':
            prob = 1 / (1 + np.exp(-y_pred))
            grad = prob - y_true
            hess = prob * (1 - prob)
            return grad, hess
        else:
            raise ValueError("Unsupported objective")

    def fit(self, X, y, X_val=None, y_val=None, early_stopping_rounds=None, eval_metric=None):
        m, n = X.shape
        is_multiclass = self.num_classes > 1
        scores = np.zeros((m, self.num_classes)) if is_multiclass else np.full(m, np.mean(y))
        best_score = float('inf')
        best_iteration = 0
        rounds_without_improvement = 0

        val_scores = None
        if X_val is not None:
            val_scores = np.zeros((X_val.shape[0], self.num_classes)) if is_multiclass else np.full(X_val.shape[0],
                                                                                                    np.mean(y))

        for i in range(self.n_estimators):
            feature_indices = np.random.choice(n, int(self.colsample_bytree * n), replace=False)
            self.feature_subsets.append(feature_indices)
            X_subset = X[:, feature_indices]
            score = None
            if is_multiclass:
                probs = self._softmax(scores)
                trees = []
                for k in range(self.num_classes):
                    y_k = (y == k).astype(float)
                    g_k = probs[:, k] - y_k
                    h_k = probs[:, k] * (1 - probs[:, k])
                    tree = self._fit_tree(X_subset, g_k, h_k, 0)
                    update = self._predict_tree(tree, X_subset)
                    scores[:, k] -= self.learning_rate * update
                    trees.append(tree)
                self.models.append(trees)

                # Early stopping
                if (X_val is not None and
                        y_val is not None and
                        eval_metric is not None and
                        val_scores is not None):
                    for k, tree in enumerate(trees):
                        update = self._predict_tree(tree, X_val[:, feature_indices])
                        val_scores[:, k] -= self.learning_rate * update
                    val_probs = self._softmax(val_scores)
                    score = eval_metric(y_val, val_probs)
            else:
                grad, hess = self._get_grad_hess(y, scores)
                tree = self._fit_tree(X_subset, grad, hess, 0)
                update = self._predict_tree(tree, X_subset)
                scores -= self.learning_rate * update
                self.models.append(tree)

                if X_val is not None and y_val is not None and eval_metric is not None:
                    update = self._predict_tree(tree, X_val[:, feature_indices])
                    val_scores -= self.learning_rate * update
                    score = eval_metric(y_val, val_scores)

            if (X_val is not None and
                    eval_metric is not None and
                    score is not None):
                if score < best_score:
                    best_score = score
                    best_iteration = i
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                    if early_stopping_rounds and rounds_without_improvement >= early_stopping_rounds:
                        print(f"Early stopping at round {i + 1}, best round was {best_iteration + 1}")
                        break

        self.best_iteration = best_iteration + 1

    def predict(self, X):
        m = X.shape[0]
        is_multiclass = self.num_classes > 1
        scores = np.zeros((m, self.num_classes)) if is_multiclass else np.zeros(m)

        for i, features in enumerate(self.feature_subsets):
            if self.best_iteration and i >= self.best_iteration:
                break
            X_subset = X[:, features]
            if is_multiclass:
                for k in range(self.num_classes):
                    scores[:, k] -= self.learning_rate * self._predict_tree(self.models[i][k], X_subset)
            else:
                scores -= self.learning_rate * self._predict_tree(self.models[i], X_subset)

        return np.argmax(self._softmax(scores), axis=1) if is_multiclass else scores

    def _fit_tree(self, X, grad, hess, depth):
        if depth >= self.max_depth or X.shape[0] <= 1:
            return self._create_leaf(grad, hess)
        best_gain = -np.inf
        best_split = None
        G_total = np.sum(grad)
        H_total = np.sum(hess)

        for feature in range(X.shape[1]):
            #  thresholds = np.unique(X[:, feature])
            thresholds = np.percentile(X[:, feature], np.linspace(0, 100, 10))  # 10 thresholds
            for threshold in thresholds:
                left = X[:, feature] <= threshold
                right = ~left
                if np.sum(left) == 0 or np.sum(right) == 0:
                    continue
                G_L = np.sum(grad[left])
                H_L = np.sum(hess[left])
                G_R = np.sum(grad[right])
                H_R = np.sum(hess[right])
                gain = 0.5 * ((G_L ** 2 / (H_L + self.reg_lambda)) +
                              (G_R ** 2 / (H_R + self.reg_lambda)) -
                              (G_total ** 2 / (H_total + self.reg_lambda))) - self.gamma
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, threshold, left, right)

        if best_gain <= 0 or best_split is None:
            return self._create_leaf(grad, hess)

        feature, threshold, left, right = best_split
        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._fit_tree(X[left], grad[left], hess[left], depth + 1),
            "right": self._fit_tree(X[right], grad[right], hess[right], depth + 1)
        }

    def _create_leaf(self, grad, hess):
        value = -np.sum(grad) / (np.sum(hess) + self.reg_lambda)
        return {"value": value}

    def _predict_tree(self, tree, X):
        if "value" in tree:
            return np.full(X.shape[0], tree["value"])
        feature = tree["feature"]
        threshold = tree["threshold"]
        left = X[:, feature] <= threshold
        right = ~left
        y_pred = np.empty(X.shape[0])
        y_pred[left] = self._predict_tree(tree["left"], X[left])
        y_pred[right] = self._predict_tree(tree["right"], X[right])
        return y_pred


# AUC metric for multiclass (macro-averaged)
def multiclass_auc(y_true, y_prob):
    K = y_prob.shape[1]
    y_onehot = np.zeros_like(y_prob)
    y_onehot[np.arange(y_true.shape[0]), y_true] = 1
    aucs = []
    for k in range(K):
        try:
            auc = roc_auc_score(y_onehot[:, k], y_prob[:, k])
            aucs.append(auc)
        except (ValueError, IndexError, TypeError, AttributeError):
            continue
    return 1 - np.mean(aucs)  # Lower = better for early stopping


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
