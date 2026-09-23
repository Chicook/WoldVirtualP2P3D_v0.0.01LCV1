class MathematicalPrecision2026:
    """Utilidades de precision matematica neuronal y algoritmos 2026 para SGD."""

    EPS: float = 1e-12

    @staticmethod
    def sign_sgd_ef21(G: np.ndarray, e: np.ndarray, lr: float) -> Tuple[np.ndarray, np.ndarray]:
        """Sign-SGD con Error Feedback 21 (EF21)."""
        d = max(G.size, 1)
        v = G + e
        p = np.sign(v)
        scale = np.sum(np.abs(v)) / d
        return lr * scale * p, v - scale * p

    @staticmethod
    def cosine_warm_restart(t: int, T_0: int = 10, T_mult: int = 2,
                            eta_min: float = 1e-5, eta_max: float = 1e-1) -> float:
        """SGDR: Tasa de aprendizaje coseno con reinicios calientes periodicos."""
        curr_T = T_0
        t_curr = t
        while t_curr >= curr_T:
            t_curr -= curr_T
            curr_T *= T_mult
        return eta_min + 0.5 * (eta_max - eta_min) * (1.0 + math.cos(math.pi * (t_curr / max(curr_T, 1))))

    @staticmethod
    def top_k_sparsify(G: np.ndarray, density: float = 0.1) -> np.ndarray:
        """Compresion de gradiente Top-K adaptativa 2026 para comunicacion distribuida."""
        k = max(1, int(G.size * density))
        flat = np.abs(G.ravel())
        threshold = np.partition(flat, -k)[-k]
        return G * (np.abs(G) >= threshold)

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
# 2. INTERNALS - motor de optimizacion SGD 2026
# ===========================================================================
