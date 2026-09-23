class MathematicalPrecision2026:
    """Utilidades de precision matematica neuronal 2026 para AdaGrad."""

    EPS: float = 1e-12

    @staticmethod
    def adagrad_norm_step(G: np.ndarray, b_t: float, lr: float, eps: float) -> Tuple[np.ndarray, float]:
        """AdaGrad-Norm (Ward 2020-2026): actualizacion escalonada por norma global."""
        g_norm_sq = float(np.sum(G ** 2))
        new_b = b_t + g_norm_sq
        step_size = lr / (math.sqrt(new_b) + eps)
        return step_size * G, new_b

    @staticmethod
    def decayed_accumulator(G_sq: np.ndarray, G: np.ndarray,
                            decay: float = 0.9995) -> np.ndarray:
        """Acumulador AdaGrad con decaimiento debil para evitar estancamiento prematuro."""
        return decay * G_sq + (G ** 2)

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
# 2. INTERNALS - motor de optimizacion AdaGrad 2026
# ===========================================================================
