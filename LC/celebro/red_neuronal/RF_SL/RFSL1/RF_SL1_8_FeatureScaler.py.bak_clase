class FeatureScaler:
    def __init__(self, method: str = "standard"):
        self.method = method if method in ("standard", "minmax", "robust") else "standard"
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> None:
        if self.method == "standard":
            self._mean = np.mean(X, axis=0)
            self._std = np.std(X, axis=0) + 1e-8
        elif self.method == "minmax":
            self._min = np.min(X, axis=0)
            self._max = np.max(X, axis=0)
        elif self.method == "robust":
            self._mean = np.median(X, axis=0)
            self._std = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0) + 1e-8

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.method == "standard":
            return (X - self._mean) / self._std
        elif self.method == "minmax":
            return (X - self._min) / (self._max - self._min + 1e-8)
        elif self.method == "robust":
            return (X - self._mean) / self._std
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


