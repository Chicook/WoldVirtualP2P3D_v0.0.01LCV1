class ModelEvaluator:
    def __init__(self):
        self._results: Dict[str, Any] = {}

    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true == y_pred))

    def precision(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary") -> float:
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        if average == "binary":
            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            fp = int(np.sum((y_pred == 1) & (y_true == 0)))
            return tp / max(1, tp + fp)
        return 0.0

    def recall(self, y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary") -> float:
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        if average == "binary":
            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            fn = int(np.sum((y_pred == 0) & (y_true == 1)))
            return tp / max(1, tp + fn)
        return 0.0

    def f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        p = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)
        return 2.0 * p * r / max(1e-8, p + r)

    def evaluate_all(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            'accuracy': self.accuracy(y_true, y_pred),
            'precision': self.precision(y_true, y_pred),
            'recall': self.recall(y_true, y_pred),
            'f1': self.f1_score(y_true, y_pred),
        }


