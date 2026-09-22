"""init_utils.py — inicializadores compartidos."""
from __future__ import annotations
import math
import numpy as np

def inicializar_he(shape, fan_in=1):
    return np.random.normal(0, math.sqrt(2.0 / max(1, fan_in)), shape).astype(np.float32)

def inicializar_xavier(shape, fan_in=1, fan_out=1):
    lim = math.sqrt(6.0 / max(1, fan_in + fan_out))
    return np.random.uniform(-lim, lim, shape).astype(np.float32)

def inicializar_lecun(shape, fan_in=1):
    return np.random.normal(0, math.sqrt(1.0 / max(1, fan_in)), shape).astype(np.float32)

def inicializar_ortogonal(shape):
    a = np.random.randn(*shape).astype(np.float64)
    q, _ = np.linalg.qr(a) if a.shape[0] >= a.shape[1] else (None, None)
    if q is not None:
        return q.reshape(shape).astype(np.float32)
    q2, _ = np.linalg.qr(a.T)
    return q2.T.reshape(shape).astype(np.float32)
