class MathematicalPrecision2026:
    """Utilidades de precision matematica neuronal 2026 para AdaDelta."""

    EPS: float = 1e-12

    @staticmethod
    def adadelta_step(G: np.ndarray, mean_sq_g: np.ndarray, mean_sq_d: np.ndarray,
                      rho: float, eps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Paso canonico AdaDelta con consistencia dimensional [Delta x] ~ [Delta x]."""
        new_sq_g = rho * mean_sq_g + (1.0 - rho) * (G ** 2)
        rms_d = np.sqrt(mean_sq_d + eps)
        rms_g = np.sqrt(new_sq_g + eps)
        delta = -(rms_d / rms_g) * G
        new_sq_d = rho * mean_sq_d + (1.0 - rho) * (delta ** 2)
        return delta, new_sq_g, new_sq_d

    @staticmethod
    def adaptive_epsilon(G: np.ndarray, base_eps: float = 1e-6) -> float:
        """Calcula epsilon adaptativo con regularizacion homotopica."""
        g_norm = float(np.linalg.norm(G))
        scale = math.sqrt(max(G.size, 1))
        return max(base_eps, (g_norm / scale) * 1e-5)

    @staticmethod
    def newton_schulz5(G: np.ndarray, steps: int = 5) -> np.ndarray:
        """Ortogonalizacion polar grado-5 Newton-Schulz (Muon 2026)."""
        assert G.ndim == 2, "newton_schulz5 requiere matriz 2D"
        X = G / (np.linalg.norm(G, ord="fro") + MathematicalPrecision2026.EPS)
        for _ in range(steps):
            X = 1.5 * X - 0.5 * X @ (X.T @ X)
        return X

    @staticmethod
    def gsnr(G: np.ndarray, G_sq: np.ndarray, t: int) -> float:
        """Gradient Signal-to-Noise Ratio adaptativo (GSNR)."""
        mean_g, mean_g2 = G / max(t, 1), G_sq / max(t, 1)
        signal = float(np.sum(mean_g ** 2))
        noise = float(np.sum(np.maximum(mean_g2 - mean_g ** 2, 0.0)))
        return signal / (noise + MathematicalPrecision2026.EPS)

    @staticmethod
    def polyak_step(P: np.ndarray, W: np.ndarray, decay: float = 0.999) -> np.ndarray:
        """Polyak-Ruppert parameter averaging con decaimiento exponencial."""
        return decay * P + (1.0 - decay) * W


# ===========================================================================
# 2. INTERNALS - motor de optimizacion AdaDelta 2026
# ===========================================================================
