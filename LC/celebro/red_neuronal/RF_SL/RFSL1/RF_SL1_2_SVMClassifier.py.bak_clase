class SVMClassifier:
    def __init__(self, kernel: str = "rbf", C: float = 1.0, gamma: float = 0.1):
        self.kernel_type = kernel
        self.C = C
        self.gamma = gamma
        self.kernel_engine = KernelEngine(kernel, gamma)
        self.trainer = SMOTrainer(C)
        self.identifier = SupportVectorIdentifier()
        self.margin_opt = MarginOptimizer()
        self._is_trained: bool = False
        self._training_time: float = 0.0

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        t0 = time.perf_counter()
        K = self.kernel_engine.gram_matrix(X)
        alphas, bias = self.trainer.optimize(K, y)
        sv_mask = alphas > self.identifier.threshold
        self._support_vectors = X[sv_mask]
        self._support_labels = y[sv_mask]
        self._alphas = alphas[sv_mask]
        self._bias = bias
        self._is_trained = True
        self._training_time = time.perf_counter() - t0
        self.identifier.identify(alphas)
        margin = self.margin_opt.compute(self.pesos_svm) if hasattr(self, 'pesos_svm') else 0.0
        return {
            'n_support_vectors': self.identifier.count(),
            'margin': margin, 'training_time': self._training_time,
            'kernel': self.kernel_type, 'C': self.C, 'converged': True,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained: raise RuntimeError("SVM no entrenado")
        K = self.kernel_engine.compute(self._support_vectors, X)
        decision = (self._alphas * self._support_labels) @ K + self._bias
        return np.sign(decision)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained: raise RuntimeError("SVM no entrenado")
        K = self.kernel_engine.compute(self._support_vectors, X)
        return (self._alphas * self._support_labels) @ K + self._bias

    @property
    def is_trained(self) -> bool: return self._is_trained

    @property
    def n_support_vectors(self) -> int: return self.identifier.count()

    @property
    def training_time(self) -> float: return self._training_time


