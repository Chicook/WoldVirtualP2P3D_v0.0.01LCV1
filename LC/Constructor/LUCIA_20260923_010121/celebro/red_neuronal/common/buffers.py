"""buffers.py — RingBuffers compartidos."""
from __future__ import annotations
import numpy as np
from typing import Any, List

class RingBuffer:
    def __init__(self, cap: int): self.cap = cap; self._d: List[Any] = []
    def append(self, x):
        self._d.append(x)
        if len(self._d) > self.cap: del self._d[0:len(self._d) - self.cap]
    def __len__(self): return len(self._d)
    def __getitem__(self, i): return self._d[i]
    def __iter__(self): return iter(self._d)
    def clear(self): self._d.clear()
    @property
    def full(self): return len(self._d) == self.cap

class RingBufferNumpy:
    def __init__(self, cap: int, dtype=np.float32):
        self.cap = cap; self._pos = 0; self._count = 0
        self._data = np.zeros((cap,), dtype=object)
    def append(self, x):
        self._data[self._pos] = x; self._pos = (self._pos + 1) % self.cap
        if self._count < self.cap: self._count += 1
    def __len__(self): return self._count
    def __getitem__(self, i):
        if i >= self._count: raise IndexError(i)
        return self._data[(self._pos - self._count + i) % self.cap]
    @property
    def full(self): return self._count == self.cap
    def clear(self): self._pos = 0; self._count = 0
