class MathematicalPrecision2026:
    """Utilidades de precision matematica neuronal de 2026."""

    EPS: float = 1e-12

    @staticmethod
    def newton_schulz5(G: np.ndarray, steps: int = 5) -> np.ndarray:
        """Ortogonalizacion Newton-Schulz grado-5 (Muon 2026). Aproxima factor U de G=USV^T sin SVD."""
        assert G.ndim == 2, "newton_schulz5 requiere matriz 2D"
        X = G / (np.linalg.norm(G, ord="fro") + MathematicalPrecision2026.EPS)
        a, b = 1.5, -0.5
        for _ in range(steps):
            X = a * X + b * X @ (X.T @ X)
        return X

    @staticmethod
    def soap_precondition(G: np.ndarray, L: np.ndarray, R: np.ndarray,
                          beta: float = 0.95) -> np.ndarray:
        """SOAP Shampoo-style: actualiza curvatura L/R con EMA y aplica P=L^{-1/2} G R^{-1/2}."""
        m, n = G.shape
        L[:] = beta * L + (1 - beta) * (G @ G.T)
        R[:] = beta * R + (1 - beta) * (G.T @ G)
        eps  = MathematicalPrecision2026.EPS
        return np.linalg.inv(L + eps * np.eye(m)) @ G @ np.linalg.inv(R + eps * np.eye(n))

    @staticmethod
    def gsnr(G: np.ndarray, G_sq: np.ndarray, t: int) -> float:
        """Gradient Signal-to-Noise Ratio (GSNR): signal / (noise + eps)."""
        g_sq_mean = G_sq / max(t, 1)
        g_mean_sq = (G / max(t, 1)) ** 2
        noise     = np.sum(np.maximum(g_sq_mean - g_mean_sq, 0.0))
        return float(np.sum(g_mean_sq) / (noise + MathematicalPrecision2026.EPS))

    @staticmethod
    def trust_ratio_clip(update: np.ndarray, param: np.ndarray,
                         clip: float = 10.0) -> np.ndarray:
        """Trust-ratio por capa con clipping hiperbolico."""
        w_norm = np.linalg.norm(param)
        u_norm = np.linalg.norm(update)
        ratio  = (w_norm / (u_norm + MathematicalPrecision2026.EPS))
        ratio  = min(ratio, clip)
        return update * ratio

    @staticmethod
    def polyak_average(avg: np.ndarray, current: np.ndarray,
                       decay: float = 0.999) -> np.ndarray:
        """EMA Polyak averaging de pesos."""
        return decay * avg + (1.0 - decay) * current


# ===========================================================================
# 2. INTERNALS — motor de optimizacion Backpropagation 2026
# ===========================================================================
