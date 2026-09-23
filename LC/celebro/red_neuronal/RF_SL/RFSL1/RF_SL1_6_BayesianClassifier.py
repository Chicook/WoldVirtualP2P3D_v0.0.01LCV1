class BayesianClassifier:
    def __init__(self, distribution: str = "gaussian", var_smoothing: float = 1e-9):
        self.distribution = distribution
        self.var_smoothing = var_smoothing
        self.prior = PriorEstimator()
        self.likelihood = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianClassifier':
        n_classes = len(np.unique(y))
        self.priors = self.prior.estimate(y, n_classes)
        self.likelihood = FeatureLikelihood(X.shape[1], n_classes, self.distribution)
        self.likelihood.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted: raise RuntimeError("No entrenado")
        n = len(X)
        n_classes = len(self.priors)
        log_probs = np.zeros((n, n_classes))
        for c in range(n_classes):
            log_probs[:, c] = np.log(self.priors[c] + 1e-8)
            for f in range(X.shape[1]):
                mean = self.likelihood.get_means()[f][c]
                var = self.likelihood.get_vars()[f][c]
                log_probs[:, c] += GaussianDistribution.log_pdf(X[:, f], mean, var)
        return np.argmax(log_probs, axis=1)


