class DataSplitter:
    def __init__(self, val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = 42):
        self.val_ratio = max(0.0, min(0.99, float(val_ratio)))
        self.test_ratio = max(0.0, min(0.99, float(test_ratio)))
        self.seed = int(seed)
        self._train_idx: Optional[np.ndarray] = None
        self._val_idx: Optional[np.ndarray] = None
        self._test_idx: Optional[np.ndarray] = None

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[Tuple[np.ndarray, ...], ...]:
        np.random.seed(self.seed)
        n = len(X)
        n_test = int(n * self.test_ratio)
        n_val = int(n * self.val_ratio)
        indices = np.random.permutation(n)
        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]
        self._train_idx = train_idx
        self._val_idx = val_idx
        self._test_idx = test_idx
        if y is not None:
            return ((X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx]))
        return ((X[train_idx],), (X[val_idx],), (X[test_idx],))

    def get_sizes(self, n: int) -> Dict[str, int]:
        n_test = int(n * self.test_ratio)
        n_val = int(n * self.val_ratio)
        return {'train': n - n_test - n_val, 'val': n_val, 'test': n_test}


