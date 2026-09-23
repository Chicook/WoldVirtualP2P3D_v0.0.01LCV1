class NeuralNetTrainer:
    def __init__(self, network: NeuralNetworkOptimizer, max_norm: float = 5.0):
        self.network = network; self.max_norm = max_norm
        self._train_losses: List[float] = []; self._val_losses: List[float] = []

    def train_epoch(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.network.forward(X)
        loss = self.network.compute_loss(pred, y)
        grad = self.network.compute_gradient(pred, y)
        grad = GradientClipper(self.max_norm).clip(grad)
        gp, gs = self.network.backward(grad, X)
        self.network.update_weights_momentum(gp, gs)
        self._train_losses.append(loss)
        return loss

    def validate(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.network.predict(X)
        loss = LossFunctions.mse(pred, y)
        self._val_losses.append(loss)
        return loss

    def train_with_validation(self, X: np.ndarray, y: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                epochs: int = 10) -> Dict[str, List[float]]:
        for epoch in range(epochs):
            train_loss = self.train_epoch(X, y)
            val_loss = self.validate(X_val, y_val)
            logger.info(f"Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}")
        return {'train': self._train_losses, 'val': self._val_losses}

    def get_train_losses(self) -> List[float]: return self._train_losses.copy()

    def get_val_losses(self) -> List[float]: return self._val_losses.copy()


