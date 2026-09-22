"""neural_math.py — Motor matemático único 2026 (Muon NS-5 + SOAP + GSNR + Shannon)."""
from __future__ import annotations
import math
import numpy as np

__all__ = ["NeuralMathPrecision2026", "SemanticEncoderSHA", "SemanticEncoder4D"]


class NeuralMathPrecision2026:
    EPS: float = 1e-12

    @staticmethod
    def newton_schulz5(G: np.ndarray, steps: int = 5) -> np.ndarray:
        if G.ndim != 2:
            return G
        X = G / (np.linalg.norm(G, ord="fro") + NeuralMathPrecision2026.EPS)
        for _ in range(steps):
            X = 1.5 * X - 0.5 * X @ (X.T @ X)
        return X

    @staticmethod
    def soap_precondition(G, L, R, beta: float = 0.95):
        m, n = G.shape
        L[:] = beta * L + (1.0 - beta) * (G @ G.T)
        R[:] = beta * R + (1.0 - beta) * (G.T @ G)
        eps = NeuralMathPrecision2026.EPS
        L_inv = np.linalg.pinv(L + eps * np.eye(m))
        R_inv = np.linalg.pinv(R + eps * np.eye(n))
        return L_inv @ G @ R_inv

    @staticmethod
    def gsnr(G: np.ndarray, G_sq: np.ndarray, t: int) -> float:
        mean_g = G / max(t, 1)
        mean_g2 = G_sq / max(t, 1)
        signal = float(np.sum(mean_g ** 2))
        noise = float(np.sum(np.maximum(mean_g2 - mean_g ** 2, 0.0)))
        return signal / (noise + NeuralMathPrecision2026.EPS)

    @staticmethod
    def trust_ratio_clip(update: np.ndarray, param: np.ndarray, clip: float = 5.0) -> np.ndarray:
        w_norm = float(np.linalg.norm(param))
        u_norm = float(np.linalg.norm(update))
        ratio = min(w_norm / (u_norm + NeuralMathPrecision2026.EPS), clip)
        return update * ratio

    @staticmethod
    def shannon_entropy(vector: np.ndarray) -> float:
        v = np.abs(np.asarray(vector).ravel())
        s = float(np.sum(v))
        if s < NeuralMathPrecision2026.EPS:
            return 0.0
        p = v / s
        p = p[p > 0]
        h = -float(np.sum(p * np.log2(p + 1e-15)))
        max_h = math.log2(len(v)) if len(v) > 1 else 1.0
        return float(np.clip(h / max(max_h, 1e-5), 0.0, 1.0))

    @staticmethod
    def to_8d(val) -> np.ndarray:
        if val is None:
            return np.zeros((1, 8), dtype=np.float32)
        if isinstance(val, (tuple, list)):
            val = val[0]
        arr = np.asarray(val, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] == 8:
            return arr
        flat = arr.ravel()
        reps = int(np.ceil(8 / max(1, flat.size)))
        return np.tile(flat, reps)[:8].reshape(1, 8).astype(np.float32)

    @staticmethod
    def ajustar_forma(mat: np.ndarray, shape: tuple) -> np.ndarray:
        """Redimensiona con crop/pad sensato en lugar de tile ciego."""
        mat = np.asarray(mat, dtype=np.float32)
        if mat.shape == shape:
            return mat
        # sesgos 1D: usa media por columnas
        if len(shape) == 2 and shape[0] == 1 and mat.ndim == 2:
            v = np.mean(mat, axis=0)
            if v.size >= shape[1]:
                return v[: shape[1]].reshape(shape)
            out = np.zeros(shape, dtype=np.float32)
            out[0, : v.size] = v
            return out
        flat = mat.ravel()
        need = int(np.prod(shape))
        if flat.size >= need:
            return flat[:need].reshape(shape).astype(np.float32)
        out = np.zeros(need, dtype=np.float32)
        out[: flat.size] = flat
        return out.reshape(shape).astype(np.float32)


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
