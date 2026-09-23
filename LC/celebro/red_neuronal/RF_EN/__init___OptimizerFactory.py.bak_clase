class OptimizerFactory:
    def __init__(self):
        self._optimizers: Dict[str, Callable] = {
            'sgd': self._sgd, 'adam': self._adam, 'rmsprop': self._rmsprop, 'adagrad': self._adagrad
        }
    def create(self, name: str, lr: float = 0.001, **kwargs) -> Callable:
        fn = self._optimizers.get(name.lower(), self._adam)
        return lambda params, grads: fn(params, grads, lr, **kwargs)
    @staticmethod
    def _sgd(params: List[np.ndarray], grads: List[np.ndarray], lr: float, momentum: float = 0.9) -> List[np.ndarray]:
        updated = []; v = [np.zeros_like(p) for p in params]
        for i, (p, g) in enumerate(zip(params, grads)):
            v[i] = momentum * v[i] + g; updated.append(p - lr * v[i])
        return updated
    @staticmethod
    def _adam(params: List[np.ndarray], grads: List[np.ndarray], lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> List[np.ndarray]:
        m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]; t = 0
        updated = []
        for i, (p, g) in enumerate(zip(params, grads)):
            t += 1; m[i] = beta1 * m[i] + (1 - beta1) * g; v[i] = beta2 * v[i] + (1 - beta2) * g * g
            mh = m[i] / (1 - beta1 ** t); vh = v[i] / (1 - beta2 ** t); updated.append(p - lr * mh / (np.sqrt(vh) + eps))
        return updated
    @staticmethod
    def _rmsprop(params: List[np.ndarray], grads: List[np.ndarray], lr: float, decay: float = 0.9, eps: float = 1e-8) -> List[np.ndarray]:
        cache = [np.zeros_like(p) for p in params]; updated = []
        for i, (p, g) in enumerate(zip(params, grads)):
            cache[i] = decay * cache[i] + (1 - decay) * g * g; updated.append(p - lr * g / (np.sqrt(cache[i]) + eps))
        return updated
    @staticmethod
    def _adagrad(params: List[np.ndarray], grads: List[np.ndarray], lr: float, eps: float = 1e-8) -> List[np.ndarray]:
        cache = [np.zeros_like(p) for p in params]; updated = []
        for i, (p, g) in enumerate(zip(params, grads)):
            cache[i] += g * g; updated.append(p - lr * g / (np.sqrt(cache[i]) + eps))
        return updated

