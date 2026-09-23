class AdaDeltaOptimizerInternal:
    """Motor interno de AdaDelta con formulacion de segundo orden + Muon (2026)."""

    def __init__(self, learning_rate: float, rho: float, eps: float,
                 weight_decay: float, use_muon: bool = True,
                 polyak_decay: float = 0.999):
        self.lr = learning_rate
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        self.use_muon = use_muon
        self.polyak_decay = polyak_decay
        self._sq_grad: Optional[np.ndarray] = None
        self._sq_delta: Optional[np.ndarray] = None
        self._polyak: Optional[np.ndarray] = None
        self._G_sum: Optional[np.ndarray] = None
        self._G_sq_sum: Optional[np.ndarray] = None
        self.step_count = 0
        self.adadelta_score = 0.0
        self.delta_score = 0.0
        self.muon_orthogonality = 0.0
        self.gsnr_score = 0.0
        self.current_eps = eps

    def _init_state(self, G: np.ndarray) -> None:
        self._sq_grad = np.zeros_like(G)
        self._sq_delta = np.zeros_like(G)
        self._polyak = np.zeros_like(G)
        self._G_sum = np.zeros_like(G)
        self._G_sq_sum = np.zeros_like(G)

    def step(self, G: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Ejecuta un paso de optimizacion AdaDelta 2026 y devuelve metricas."""
        self.step_count += 1
        t = self.step_count
        if G is None:
            G = np.random.randn(8, 8).astype(np.float64) * 0.1
        if self._sq_grad is None:
            self._init_state(G)

        self._G_sum += G
        self._G_sq_sum += G ** 2

        self.current_eps = MathematicalPrecision2026.adaptive_epsilon(G, base_eps=self.eps)
        grad = G + self.weight_decay * self._polyak if self.weight_decay > 0 else G.copy()

        delta, self._sq_grad, self._sq_delta = MathematicalPrecision2026.adadelta_step(
            grad, self._sq_grad, self._sq_delta, self.rho, self.current_eps
        )
        step_grad = -delta

        if self.use_muon and step_grad.ndim == 2:
            muon_ortho = MathematicalPrecision2026.newton_schulz5(step_grad, steps=5)
            expected = math.sqrt(min(step_grad.shape))
            self.muon_orthogonality = max(0.0, 1.0 - abs(np.linalg.norm(muon_ortho, ord="fro") - expected) / max(expected, 1.0))
            step_grad = 0.5 * step_grad + 0.5 * muon_ortho

        self._polyak = MathematicalPrecision2026.polyak_step(self._polyak, step_grad, decay=self.polyak_decay)
        self.gsnr_score = MathematicalPrecision2026.gsnr(self._G_sum, self._G_sq_sum, t)

        std_v, mean_v = float(np.std(step_grad)), float(np.mean(np.abs(step_grad))) + 1e-8
        self.adadelta_score = max(0.0, min(1.0, 1.0 - (std_v / (mean_v * 3.0))))
        self.delta_score = max(0.0, min(1.0, float(np.tanh(self.gsnr_score * 0.5))))

        return {
            "adadelta_score": self.adadelta_score,
            "delta_score": self.delta_score,
            "muon_orthogonality": self.muon_orthogonality,
            "gsnr": self.gsnr_score,
            "effective_eps": self.current_eps,
        }


# ===========================================================================
# 3. OPTIMIZADOR PRINCIPAL - AdaDeltaOptimizer
# ===========================================================================
