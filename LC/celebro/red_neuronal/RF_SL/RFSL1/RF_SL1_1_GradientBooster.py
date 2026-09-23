class GradientBooster:
    def __init__(self, input_size: int = 4, output_size: int = 8, n_estimators: int = 50,
                 learning_rate: float = 0.1, max_depth: int = 3, subsample: float = 1.0):
        self.input_size = input_size
        self.output_size = output_size
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self._trees: List[TreeBuilder] = []
        self._loss_fn = LossCalculator("mse")
        self._sampler = Sampler(subsample)
        self._aggregator = PredictionAggregator(learning_rate)
        self._residuals: Optional[np.ndarray] = None
        self._training_history: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        self._residuals = y.copy().astype(np.float64)
        self._training_history.clear()
        for i in range(self.n_estimators):
            X_s, r_s = self._sampler.sample(X, self._residuals)
            tree = TreeBuilder(max_depth=self.max_depth)
            tree.fit(X_s, r_s)
            pred = tree.predict(X)
            self._trees.append(tree)
            self._aggregator.add_prediction(pred)
            self._residuals = self._residuals - self.learning_rate * pred
            loss = self._loss_fn.compute(self._aggregator.aggregate(), y)
            self._training_history.append(loss)
        return self._training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._aggregator.aggregate() if self._aggregator.n_predictions > 0 else np.zeros(len(X))

    def get_n_estimators(self) -> int: return len(self._trees)

    def get_trees_info(self) -> List[Dict]:
        return [{'tree_idx': i, 'n_predictions': self._aggregator.n_predictions} for i in range(len(self._trees))]


