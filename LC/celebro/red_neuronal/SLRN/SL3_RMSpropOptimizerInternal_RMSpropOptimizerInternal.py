class RMSpropOptimizerInternal:
    """Motor interno de RMSprop con Centered RMS + Nesterov + Muon (2026)."""

    def __init__(self, learning_rate: float, alpha: float, eps: float,
                 weight_decay: float, momentum: float = 0.9,
                 centered: bool = True, use_muon: bool = True,
                 polyak_decay: float = 0.999):
        self.lr = learning_rate
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        self.use_muon = use_muon
        self.polyak_decay = polyak_decay
        self._mean_sq: Optional[np.ndarray] = None
        self._mean_g: Optional[np.ndarray] = None
        self._velocity: Optional[np.ndarray] = None
        self._polyak: Optional[np.ndarray] = None
        self._G_sum: Optional[np.ndarray] = None
        self._G_sq: Optional[np.ndarray] = None
        self.step_count = 0
        self.rmsprop_score = 0.0
        self.rms_score = 0.0
        self.muon_orthogonality = 0.0
        self.gsnr_score = 0.0
        self.current_eps = eps

    def _init_state(self, G: np.ndarray) -> None:
        self._mean_sq = np.zeros_like(G)
        self._mean_g = np.zeros_like(G)
        self._velocity = np.zeros_like(G)
        self._polyak = np.zeros_like(G)
        self._G_sum = np.zeros_like(G)
        self._G_sq = np.zeros_like(G)

    def step(self, G: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Ejecuta un paso de optimizacion RMSprop 2026 y devuelve metricas."""
        self.step_count += 1
        t = self.step_count
        if G is None:
            G = np.random.randn(8, 8).astype(np.float64) * 0.1
        if self._mean_sq is None:
            self._init_state(G)

        self._G_sum += G
        self._G_sq += G ** 2

        self.current_eps = MathematicalPrecision2026.adaptive_epsilon(G, base_eps=self.eps)
        grad = G + self.weight_decay * self._polyak if self.weight_decay > 0 else G.copy()

        if self.centered:
            scaled_g, self._mean_g, self._mean_sq = MathematicalPrecision2026.centered_rms_update(
                grad, self._mean_g, self._mean_sq, self.alpha, self.current_eps
            )
        else:
            self._mean_sq = self.alpha * self._mean_sq + (1.0 - self.alpha) * (grad ** 2)
            scaled_g = grad / (np.sqrt(self._mean_sq) + self.current_eps)

        if self.momentum > 0:
            self._velocity = self.momentum * self._velocity + scaled_g
            step_grad = self._velocity.copy()
        else:
            step_grad = scaled_g

        if self.use_muon and step_grad.ndim == 2:
            muon_ortho = MathematicalPrecision2026.newton_schulz5(step_grad, steps=5)
            expected = math.sqrt(min(step_grad.shape))
            self.muon_orthogonality = max(0.0, 1.0 - abs(np.linalg.norm(muon_ortho, ord="fro") - expected) / max(expected, 1.0))
            step_grad = 0.5 * step_grad + 0.5 * muon_ortho

        self._polyak = MathematicalPrecision2026.polyak_step(self._polyak, step_grad, decay=self.polyak_decay)
        self.gsnr_score = MathematicalPrecision2026.gsnr(self._G_sum, self._G_sq, t)

        std_v, mean_v = float(np.std(step_grad)), float(np.mean(np.abs(step_grad))) + 1e-8
        self.rmsprop_score = max(0.0, min(1.0, 1.0 - (std_v / (mean_v * 3.0))))
        self.rms_score = max(0.0, min(1.0, float(np.tanh(self.gsnr_score * 0.5))))

        return {
            "rmsprop_score": self.rmsprop_score,
            "rms_score": self.rms_score,
            "muon_orthogonality": self.muon_orthogonality,
            "gsnr": self.gsnr_score,
            "effective_eps": self.current_eps,
        }


# ===========================================================================
# 3. OPTIMIZADOR PRINCIPAL - RMSpropOptimizer
# ===========================================================================
