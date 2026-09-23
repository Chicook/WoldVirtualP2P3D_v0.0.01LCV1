class LossCalculator:
    def __init__(self, loss_type: str = "mse"):
        self.loss_type = loss_type

    def compute(self, pred: np.ndarray, target: np.ndarray) -> float:
        if self.loss_type == "mse": return float(np.mean((pred - target) ** 2))
        if self.loss_type == "mae": return float(np.mean(np.abs(pred - target)))
        if self.loss_type == "huber":
            delta = 1.0; abs_err = np.abs(pred - target)
            quadratic = np.minimum(abs_err, delta)
            linear = abs_err - quadratic
            return float(np.mean(0.5 * quadratic ** 2 + delta * linear))
        if self.loss_type == "log":
            p = np.clip(pred, 1e-8, 1 - 1e-8)
            return float(-np.mean(target * np.log(p) + (1 - target) * np.log(1 - p)))
        if self.loss_type == "smoothed_l1":
            return self.huber_loss(pred, target)
        return float(np.mean((pred - target) ** 2))

    def gradient(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        diff = pred - target
        if self.loss_type == "mse": return diff
        if self.loss_type == "huber":
            delta = 1.0; abs_diff = np.abs(diff)
            return np.where(abs_diff <= delta, diff, delta * np.sign(diff))
        if self.loss_type == "log":
            return (pred - target) / np.clip(pred * (1 - pred), 1e-8, None)
        return diff

    @staticmethod
    def huber_loss(pred: np.ndarray, target: np.ndarray, delta: float = 1.0) -> float:
        abs_err = np.abs(pred - target)
        quadratic = np.minimum(abs_err, delta)
        linear = abs_err - quadratic
        return float(np.mean(0.5 * quadratic ** 2 + delta * linear))

    @staticmethod
    def masked_loss(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
        if mask.sum() == 0: return 0.0
        return float(np.mean((pred[mask] - target[mask]) ** 2))


