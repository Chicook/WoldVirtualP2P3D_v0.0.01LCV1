class ActivationEngine:
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray: return np.maximum(0, x)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray: return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray: return np.tanh(x)

    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray: return np.where(x > 0, x, alpha * x)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=-1, keepdims=True)); return e / np.sum(e, axis=-1, keepdims=True)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray: return (x > 0).astype(np.float32)

    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray: s = ActivationEngine.sigmoid(x); return s * (1 - s)

    @staticmethod
    def apply(x: np.ndarray, activation: str) -> np.ndarray:
        if activation == "relu": return ActivationEngine.relu(x)
        if activation == "sigmoid": return ActivationEngine.sigmoid(x)
        if activation == "tanh": return ActivationEngine.tanh(x)
        if activation == "leaky_relu": return ActivationEngine.leaky_relu(x)
        if activation == "softmax": return ActivationEngine.softmax(x)
        return x


