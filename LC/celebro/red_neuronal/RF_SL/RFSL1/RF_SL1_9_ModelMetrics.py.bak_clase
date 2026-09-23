class ModelMetrics:
    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        return 1.0 - ss_res / max(1e-8, ss_tot)

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mask = y_true != 0
        if not np.any(mask): return float('inf')
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

    @staticmethod
    def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            'r2': ModelMetrics.r2(y_true, y_pred),
            'mae': ModelMetrics.mae(y_true, y_pred),
            'rmse': ModelMetrics.rmse(y_true, y_pred),
            'mape': ModelMetrics.mape(y_true, y_pred),
        }


