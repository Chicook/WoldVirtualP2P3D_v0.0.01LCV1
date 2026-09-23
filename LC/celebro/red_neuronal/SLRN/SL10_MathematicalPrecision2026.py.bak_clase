class MathematicalPrecision2026:
    """Utilidades matematicas de ensamble y optimizacion neuronal 2026."""
    EPS: float = 1e-12
    @staticmethod
    def softmax_weights(losses: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Ponderacion Softmax inversa para meta-ensamble de optimizadores."""
        scaled = - (losses - np.min(losses)) / max(temperature, MathematicalPrecision2026.EPS)
        exp_vals = np.exp(np.clip(scaled, -50.0, 50.0))
        return exp_vals / (np.sum(exp_vals) + MathematicalPrecision2026.EPS)
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
        """Gradient Signal-to-Noise Ratio adaptativo del sistema integrado."""
        mean_g = G / max(t, 1)
        mean_g2 = G_sq / max(t, 1)
        signal = float(np.sum(mean_g ** 2))
        noise = float(np.sum(np.maximum(mean_g2 - mean_g ** 2, 0.0)))
        return signal / (noise + MathematicalPrecision2026.EPS)
    @staticmethod
    def polyak_step(P: np.ndarray, W: np.ndarray, decay: float = 0.999) -> np.ndarray:
        """Polyak-Ruppert parameter averaging con decaimiento exponencial."""
        return decay * P + (1.0 - decay) * W
    @staticmethod
    def cosine_decay(step: int, max_steps: int, min_ratio: float = 0.01) -> float:
        """Decaimiento cosenoidal suave para tasa de aprendizaje."""
        progress = min(max(step / max(max_steps, 1), 0.0), 1.0)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))
# ===========================================================================
# 2. MOTOR INTERNO DEL SISTEMA INTEGRADO 2026
# ===========================================================================
