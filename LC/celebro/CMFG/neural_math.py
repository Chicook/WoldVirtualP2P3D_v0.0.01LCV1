"""neural_math.py — Motor matemático único 2026 (Muon NS-5 + SOAP + GSNR + Shannon)."""
from __future__ import annotations
import math
import numpy as np

__all__ = ["NeuralMathPrecision2026", "SemanticEncoderSHA", "SemanticEncoder4D"]


class SemanticEncoderSHA:
    """Encoder SHA-256 → vector normalizado de dim N (default 32), baja colisión."""

    def __init__(self, dim: int = 32):
        self.dim = int(dim)

    def encode(self, text: str) -> np.ndarray:
        import hashlib
        s = (text or "").strip()
        if not s:
            return np.zeros((1, self.dim), dtype=np.float32)
        # tokens -> hash por token, promediado (tipo TF ligero)
        toks = s.lower().split()
        acc = np.zeros(self.dim, dtype=np.float64)
        for tok in toks[:256]:
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            vals = np.frombuffer((h * ((self.dim // 32) + 1))[: self.dim], dtype=np.uint8).astype(np.float64)
            acc += (vals / 127.5 - 1.0)
        # longitud + mayúsculas como rasgos extra mezclados
        acc[0] += (len(s) % 100) / 100.0
        acc[1] += sum(1 for c in s if c.isupper()) / max(1.0, len(s))
        acc /= max(1, min(len(toks), 256))
        vec = acc.reshape(1, -1).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)

    def encode_pair(self, prompt: str, response: str) -> np.ndarray:
        comb = 0.4 * self.encode(prompt) + 0.6 * self.encode(response)
        return comb / (np.linalg.norm(comb) + 1e-8)


class SemanticEncoder4D(SemanticEncoderSHA):
    """Compatibilidad legacy: proyecta encoder SHA-32D a 4D."""

    def __init__(self):
        super().__init__(dim=32)
        rng = np.random.default_rng(2026)
        self._proj = rng.standard_normal((32, 4)).astype(np.float32) * 0.25

    def encode(self, text: str):  # type: ignore[override]
        v32 = super().encode(text)
        v4 = v32 @ self._proj
        return v4 / (np.linalg.norm(v4) + 1e-8)

    def encode_pair(self, prompt: str, response: str):  # type: ignore[override]
        comb = 0.4 * self.encode(prompt) + 0.6 * self.encode(response)
        return comb / (np.linalg.norm(comb) + 1e-8)
from LC.celebro.CMFG.neural_math_NeuralMathPrecision2026 import NeuralMathPrecision2026  # CLASSPACK
