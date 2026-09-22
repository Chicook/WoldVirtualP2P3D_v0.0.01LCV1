from __future__ import annotations
import numpy as np, math, time
from typing import *
class PerformanceProfiler:
    def __init__(self, enabled: bool = LUCIA_RL_CONFIG['profiler_enabled']):
        self._enabled = enabled; self._timings: Dict[str, List[float]] = {}; self._start_times: Dict[str, float] = {}
    def start(self, label: str) -> None:
        if not self._enabled: return
        self._start_times[label] = time.perf_counter()
    def stop(self, label: str) -> float:
        if not self._enabled or label not in self._start_times: return 0.0
        elapsed = time.perf_counter() - self._start_times.pop(label)
        if label not in self._timings: self._timings[label] = []
        self._timings[label].append(elapsed); return elapsed
    def get_avg(self, label: str) -> float:
        if label not in self._timings or not self._timings[label]: return 0.0
        return float(np.mean(self._timings[label]))
    def get_all_avg(self) -> Dict[str, float]: return {k: self.get_avg(k) for k in self._timings}
    def reset(self) -> None: self._timings.clear(); self._start_times.clear()

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

class WeightStabilityTracker:
    def __init__(self, window: int = 100):
        self._window = window; self._norms: deque = deque(maxlen=window)
    def record(self, weights: np.ndarray) -> None: self._norms.append(float(np.linalg.norm(weights)))
    def get_stability(self) -> float:
        if len(self._norms) < 2: return 1.0
        diffs = np.diff(list(self._norms)); return float(1.0 / (1.0 + np.std(diffs)))
    def get_variance(self) -> float:
        if len(self._norms) == 0:
            return 0.0
        return float(np.var(list(self._norms)))
    def is_diverging(self, threshold: float = 10.0) -> bool:
        if len(self._norms) < 10: return False
        recent = list(self._norms)[-10:]
        return max(recent) / max(1e-8, min(recent)) > threshold

class BatchPreprocessor:
    def __init__(self): self._normalizer = None; self._running_mean = 0.0; self._running_var = 1.0; self._count = 0
    def update(self, batch: np.ndarray) -> None:
        bs = np.mean(batch); bv = np.var(batch); self._count += 1
        alpha = 1.0 / self._count; self._running_mean = (1-alpha)*self._running_mean + alpha*bs; self._running_var = (1-alpha)*self._running_var + alpha*bv
    def normalize(self, batch: np.ndarray) -> np.ndarray: return (batch - self._running_mean) / max(1e-8, np.sqrt(self._running_var))
    def denormalize(self, batch: np.ndarray) -> np.ndarray: return batch * max(1e-8, np.sqrt(self._running_var)) + self._running_mean
    def should_normalize(self) -> bool: return self._count > 5

class TensorAccelerator:
    @staticmethod
    def fast_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray: return np.dot(a.astype(np.float32), b.astype(np.float32))
    @staticmethod
    def batch_dot(vectors: np.ndarray, matrix: np.ndarray) -> np.ndarray: return vectors @ matrix.T
    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=axis, keepdims=True)); return e / np.sum(e, axis=axis, keepdims=True)
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray: return np.maximum(0, x)
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray: return (x > 0).astype(np.float32)
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray: return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray: return np.tanh(x)
    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray: return np.where(x > 0, x, alpha * x)
    @staticmethod
    def clip(x: np.ndarray, min_val: float, max_val: float) -> np.ndarray: return np.clip(x, min_val, max_val)
    @staticmethod
    def l2_regularize(weights: np.ndarray, lambda_: float = 0.001) -> np.ndarray: return lambda_ * weights
    @staticmethod
    def l1_regularize(weights: np.ndarray, lambda_: float = 0.001) -> np.ndarray: return lambda_ * np.sign(weights)

class WeightScheduler:
    def __init__(self, initial_lr: float, strategy: str = 'cosine', total_steps: int = 1000):
        self._lr = initial_lr; self._strategy = strategy; self._total = total_steps; self._step = 0
    def step(self) -> float:
        self._step += 1; t = self._step / self._total
        if self._strategy == 'cosine': self._lr = 0.5 * self._lr * (1 + math.cos(math.pi * t))
        elif self._strategy == 'linear': self._lr = self._lr * (1 - t)
        elif self._strategy == 'step': self._lr = self._lr * (0.5 ** (self._step // 100))
        return self._lr
    def get_lr(self) -> float: return self._lr
    def reset(self) -> None: self._step = 0; self._lr = None if not hasattr(self, '_lr') else self._lr

class ConvergenceEngine:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4, window: int = 50):
        self._patience = patience; self._delta = min_delta; self._window = window; self._best = float('inf'); self._counter = 0; self._history: List[float] = []
    def update(self, value: float) -> bool:
        self._history.append(value); improved = value < self._best - self._delta
        if improved: self._best = value; self._counter = 0; return False
        self._counter += 1; return self._counter >= self._patience
    def is_converged(self) -> bool: return self._counter >= self._patience
    def get_improvement_rate(self) -> float:
        if len(self._history) < 2: return 0.0
        d = self._history[-1] - self._history[0]
        return float(abs(d) / max(1e-8, len(self._history)))
    def reset(self) -> None: self._best = float('inf'); self._counter = 0; self._history.clear()

class MemoryEfficientBuffer:
    def __init__(self, capacity: int, dtype: np.dtype = np.float32):
        self.cap = capacity; self.pos = 0; self.count = 0
        self.data = np.zeros((capacity,), dtype=object)
    def push(self, item) -> None:
        self.data[self.pos] = item; self.pos = (self.pos + 1) % self.cap
        if self.count < self.cap: self.count += 1
    def sample_batch(self, batch_size: int) -> List:
        n = min(batch_size, self.count); idx = np.random.choice(self.count, size=n, replace=False)
        return [self.data[i] for i in idx]
    def __len__(self) -> int: return self.count
    def is_full(self) -> bool: return self.count == self.cap
    def clear(self) -> None: self.pos = 0; self.count = 0; self.data = np.zeros((self.cap,), dtype=object)

