class DecisionTree:
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self._tree: Optional[Dict] = None
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        self._tree = self._build_tree(X, y, depth=0)
        self.feature_importances_ = np.zeros(X.shape[1])
        self._compute_importances(self._tree, X.shape[1])
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Dict:
        n = len(X)
        if depth >= self.max_depth or n < self.min_samples_split or len(np.unique(y)) == 1:
            return {'leaf': True, 'value': float(np.mean(y))}
        feat, thresh, gain = self._best_split(X, y)
        if gain < 1e-8: return {'leaf': True, 'value': float(np.mean(y))}
        left_mask = X[:, feat] <= thresh
        return {
            'leaf': False, 'feature': feat, 'threshold': thresh, 'gain': float(gain),
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[~left_mask], y[~left_mask], depth + 1),
        }

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float, float]:
        best_f, best_t, best_g = 0, 0.0, 0.0
        parent_var = float(np.var(y))
        for f in range(X.shape[1]):
            for t in np.unique(X[:, f]):
                lm = X[:, f] <= t
                if lm.sum() == 0 or lm.sum() == len(y): continue
                lv = float(np.var(y[lm])); rv = float(np.var(y[~lm]))
                n_l, n_r = lm.sum(), (~lm).sum()
                gain = parent_var - (n_l * lv + n_r * rv) / len(y)
                if gain > best_g: best_g = gain; best_f = f; best_t = float(t)
        return best_f, best_t, best_g

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._pred_one(x, self._tree) for x in X])

    def _pred_one(self, x: np.ndarray, node: Dict) -> float:
        if node['leaf']: return node['value']
        if x[node['feature']] <= node['threshold']: return self._pred_one(x, node['left'])
        return self._pred_one(x, node['right'])

    def _compute_importances(self, node: Dict, n_features: int) -> None:
        if node.get('leaf'): return
        if 'feature' in node:
            self.feature_importances_[node['feature']] += node.get('gain', 0)
            self._compute_importances(node.get('left', {}), n_features)
            self._compute_importances(node.get('right', {}), n_features)


