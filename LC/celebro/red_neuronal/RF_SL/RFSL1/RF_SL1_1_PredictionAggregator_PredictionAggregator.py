class PredictionAggregator:
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self._predictions: List[np.ndarray] = []
        self._weights: List[float] = []

    def add_prediction(self, prediction: np.ndarray, weight: float = 1.0) -> None:
        self._predictions.append(prediction)
        self._weights.append(weight)

    def aggregate(self) -> np.ndarray:
        if not self._predictions: return np.zeros(1)
        result = np.zeros_like(self._predictions[0], dtype=np.float64)
        for pred, w in zip(self._predictions, self._weights):
            result += self.learning_rate * w * pred.astype(np.float64)
        return result

    def weighted_aggregate(self, weights: List[float]) -> np.ndarray:
        if not self._predictions: return np.zeros(1)
        result = np.zeros_like(self._predictions[0], dtype=np.float64)
        for pred, w in zip(self._predictions, weights):
            result += self.learning_rate * w * pred.astype(np.float64)
        return result

    def reset(self) -> None:
        self._predictions.clear()
        self._weights.clear()

    @property
    def n_predictions(self) -> int: return len(self._predictions)


