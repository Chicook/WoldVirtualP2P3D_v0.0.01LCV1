class MathematicalPrecision2026:
    """Utilidades de precision matematica neuronal 2026 para optimizadores RMS."""

    EPS: float = 1e-12

    @staticmethod
    def centered_rms_update(G: np.ndarray, mean_g: np.ndarray, mean_sq: np.ndarray,
                            alpha: float, eps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcula RMS centrado (Graves) estimando media y varianza desacopladas."""
        new_mean_g = alpha * mean_g + (1.0 - alpha) * G
        new_mean_sq = alpha * mean_sq + (1.0 - alpha) * (G ** 2)
        variance = np.maximum(new_mean_sq - (new_mean_g ** 2), 0.0)
        scaled_g = G / (np.sqrt(variance) + eps)
        return scaled_g, new_mean_g, new_mean_sq

    @staticmethod
    def adaptive_epsilon(G: np.ndarray, base_eps: float = 1e-8) -> float:
        """Epsilon adaptativo segun la escala y cota espectral del gradiente."""
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
# 2. INTERNALS - motor de optimizacion RMSprop 2026
# ===========================================================================
