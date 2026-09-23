class TreeBuilder:
    def __init__(self, max_depth: int = 3, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self._tree: Optional[Dict] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict:
        self._tree = self._build_tree(X, y, depth=0)
        return self._tree

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Dict:
        n_samples = len(X)
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return {'leaf': True, 'value': float(np.mean(y))}
        best_feat, best_thresh, best_gain = self._find_best_split(X, y)
        if best_gain < 1e-8: return {'leaf': True, 'value': float(np.mean(y))}
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask
        return {
            'leaf': False, 'feature': best_feat, 'threshold': best_thresh,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1),
            'gain': float(best_gain), 'n_samples': n_samples,
        }

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float, float]:
        best_feat, best_thresh, best_gain = 0, 0.0, 0.0
        parent_var = float(np.var(y))
        n_features = X.shape[1]
        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                if left_mask.sum() == 0 or left_mask.sum() == len(y): continue
                left_var = float(np.var(y[left_mask]))
                right_var = float(np.var(y[~left_mask]))
                n_l, n_r = left_mask.sum(), (~left_mask).sum()
                gain = parent_var - (n_l * left_var + n_r * right_var) / len(y)
                if gain > best_gain: best_gain = gain; best_feat = feat; best_thresh = float(thresh)
        return best_feat, best_thresh, best_gain

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._tree is None: raise RuntimeError("Tree no ajustado")
        return np.array([self._predict_one(x, self._tree) for x in X])

    def _predict_one(self, x: np.ndarray, node: Dict) -> float:
        if node['leaf']: return node['value']
        if x[node['feature']] <= node['threshold']: return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])


