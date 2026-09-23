class RandomForestTrainer:
    def __init__(self, n_trees: int = 100, max_depth: int = 10,
                 subsample_ratio: float = 0.8, n_features: Optional[int] = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.subsample_ratio = subsample_ratio
        self.n_features = n_features
        self.sampler = BootstrapSampler(self.subsample_ratio)
        self.bagger = FeatureBagger(self.n_features)
        self.voter = PredictionVoter()
        self._trees: List[DecisionTree] = []
        self._history: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        self._trees.clear()
        self._history.clear()
        n = len(X)
        for i in range(self.n_trees):
            Xs, ys, oob = self.sampler.sample(X, y)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(Xs, ys)
            self._trees.append(tree)
            pred = tree.predict(X)
            self.voter.vote(pred.flatten() if pred.ndim == 1 else pred[:, 0])
            loss = float(np.mean((pred.flatten() - y.flatten()) ** 2))
            self._history.append(loss)
        return self._history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.voter.majority_vote() if self.voter.n_votes > 0 else np.zeros(len(X))

    def get_trees(self) -> List[DecisionTree]: return self._trees.copy()

    @property
    def n_trees_built(self) -> int: return len(self._trees)

    @property
    def history(self) -> List[float]: return self._history.copy()


