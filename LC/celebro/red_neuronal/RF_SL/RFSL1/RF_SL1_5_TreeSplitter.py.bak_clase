class TreeSplitter:
    def __init__(self, criterion: str = "gini"):
        self.criterion = criterion

    def find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float, float]:
        if self.criterion == "gini": score_fn = self._gini_score
        elif self.criterion == "entropy": score_fn = self._entropy_score
        else: score_fn = self._mse_score
        best_f, best_t, best_g = 0, 0.0, 0.0
        parent = score_fn(y)
        for f in range(X.shape[1]):
            for t in np.unique(X[:, f]):
                lm = X[:, f] <= t
                if lm.sum() == 0 or lm.sum() == len(y): continue
                g = parent - (lm.sum() * score_fn(y[lm]) + (~lm).sum() * score_fn(y[~lm])) / len(y)
                if g > best_g: best_g = g; best_f = f; best_t = float(t)
        return best_f, best_t, best_g

    @staticmethod
    def _gini_score(y: np.ndarray) -> float:
        p = np.bincount(y.astype(int)) / max(1, len(y)); return float(1.0 - np.sum(p ** 2))

    @staticmethod
    def _entropy_score(y: np.ndarray) -> float:
        p = np.bincount(y.astype(int)) / max(1, len(y)); p = p[p > 0]; return float(-np.sum(p * np.log2(p)))

    @staticmethod
    def _mse_score(y: np.ndarray) -> float: return float(np.var(y))


