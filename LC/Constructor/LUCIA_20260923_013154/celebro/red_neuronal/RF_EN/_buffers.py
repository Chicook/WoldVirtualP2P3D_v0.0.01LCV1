from __future__ import annotations
import numpy as np
from collections import deque
from typing import *
class _Ring:
    def __init__(self, cap: int): self.cap = cap; self._d: List[Any] = []
    def append(self, x) -> None:
        self._d.append(x)
        if len(self._d) > self.cap: del self._d[0:len(self._d)-self.cap]
    def __len__(self) -> int: return len(self._d)
    def __getitem__(self, i: int) -> Any: return self._d[i]
    def __iter__(self): return iter(self._d)
    def clear(self) -> None: self._d.clear()
    @property
    def full(self) -> bool: return len(self._d) == self.cap

class _RingBufferNumpy:
    def __init__(self, cap: int, dtype: np.dtype = np.float32):
        self.cap = cap; self._pos = 0; self._count = 0
        self._data = np.zeros((cap,), dtype=object); self._dtype = dtype
    def append(self, x) -> None:
        self._data[self._pos] = x; self._pos = (self._pos + 1) % self.cap
        if self._count < self.cap: self._count += 1
    def __len__(self) -> int: return self._count
    def __getitem__(self, i: int) -> Any:
        if i >= self._count: raise IndexError(i); return self._data[(self._pos - self._count + i) % self.cap]
    def to_array(self) -> np.ndarray: return np.array([self._data[(self._pos - self._count + i) % self.cap] for i in range(self._count)], dtype=self._dtype)
    def sample(self, n: int) -> List[Any]:
        if n >= self._count: return list(self._data[:self._count])
        idx = np.random.choice(self._count, size=n, replace=False); return [self._data[i] for i in idx]
    def clear(self) -> None: self._pos = 0; self._count = 0; self._data = np.zeros((self.cap,), dtype=object)
    @property
    def full(self) -> bool: return self._count == self.cap

class GradientAccumulator:
    def __init__(self, n_accumulate: int = 4):
        self._n = n_accumulate; self._count = 0; self._grads: Dict[str, np.ndarray] = {}
    def add(self, grads: Dict[str, np.ndarray]) -> None:
        for k, v in grads.items():
            if k not in self._grads: self._grads[k] = np.zeros_like(v)
            self._grads[k] += v
        self._count += 1
    def ready(self) -> bool: return self._count >= self._n
    def get(self) -> Dict[str, np.ndarray]:
        return {k: v / self._n for k, v in self._grads.items()}
    def reset(self) -> None: self._count = 0; self._grads.clear()

class LRUSymbolCache:
    def __init__(self, max_size: int = LUCIA_RL_CONFIG['cache_max_size']):
        self._max = max_size; self._cache: Dict[str, Any] = {}; self._order: deque = deque()
    def get(self, key: str) -> Any:
        if key in self._cache: self._order.remove(key); self._order.append(key); return self._cache[key]
        return None
    def put(self, key: str, value: Any) -> None:
        if key in self._cache: self._order.remove(key)
        elif len(self._cache) >= self._max: oldest = self._order.popleft(); del self._cache[oldest]
        self._cache[key] = value; self._order.append(key)
    def invalidate(self, key: str) -> None:
        if key in self._cache: del self._cache[key]; self._order.remove(key)
    def clear(self) -> None: self._cache.clear(); self._order.clear()
    @property
    def size(self) -> int: return len(self._cache)

