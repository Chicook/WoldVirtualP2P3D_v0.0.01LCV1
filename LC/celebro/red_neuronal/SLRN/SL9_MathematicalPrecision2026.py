class MathematicalPrecision2026:
    """Utilidades de precision matematica neuronal 2026 para AdaBound."""

    EPS: float = 1e-12

    @staticmethod
    def adabound_step(G: np.ndarray, m: np.ndarray, v: np.ndarray,
                      beta1: float, beta2: float, lr: float, final_lr: float,
                      gamma: float, eps: float, t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcula el paso AdaBound con clipping dinamico entre cotas inferior y superior."""
        new_m = beta1 * m + (1.0 - beta1) * G
        new_v = beta2 * v + (1.0 - beta2) * (G ** 2)
        bias_corr1 = 1.0 - (beta1 ** t)
        bias_corr2 = 1.0 - (beta2 ** t)
        m_hat = new_m / max(bias_corr1, MathematicalPrecision2026.EPS)
        v_hat = new_v / max(bias_corr2, MathematicalPrecision2026.EPS)

        # Limites dinamicos dependientes del paso t
        lower_bound = final_lr * (1.0 - 1.0 / (gamma * t + 1.0))
        upper_bound = final_lr * (1.0 + 1.0 / (gamma * t))
        step_size = lr / (np.sqrt(v_hat) + eps)
        bounded_step_size = np.clip(step_size, lower_bound, upper_bound)
        step = bounded_step_size * m_hat
        return step, new_m, new_v

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
# 2. INTERNALS - motor de optimizacion AdaBound 2026
# ===========================================================================
