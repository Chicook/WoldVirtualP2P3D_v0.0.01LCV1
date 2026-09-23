class AdaBoundOptimizerInternal:
    """Motor interno de AdaBound con clipping dinamico + Muon + EMA (2026)."""

    def __init__(self, learning_rate: float, final_lr: float, gamma: float,
                 weight_decay: float, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, use_muon: bool = True, polyak_decay: float = 0.999):
        self.lr = learning_rate
        self.final_lr = final_lr
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.use_muon = use_muon
        self.polyak_decay = polyak_decay
        self._m: Optional[np.ndarray] = None
        self._v: Optional[np.ndarray] = None
        self._polyak: Optional[np.ndarray] = None
        self._G_sum: Optional[np.ndarray] = None
        self._G_sq_sum: Optional[np.ndarray] = None
        self.step_count = 0
        self.adabound_score = 0.0
        self.boundary_score = 0.0
        self.muon_orthogonality = 0.0
        self.gsnr_score = 0.0

    def _init_state(self, G: np.ndarray) -> None:
        self._m = np.zeros_like(G)
        self._v = np.zeros_like(G)
        self._polyak = np.zeros_like(G)
        self._G_sum = np.zeros_like(G)
        self._G_sq_sum = np.zeros_like(G)

    def step(self, G: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Ejecuta un paso de optimizacion AdaBound 2026 y devuelve metricas."""
        self.step_count += 1
        t = self.step_count
        if G is None:
            G = np.random.randn(8, 8).astype(np.float64) * 0.1
        if self._m is None:
            self._init_state(G)

        self._G_sum += G; self._G_sq_sum += G ** 2
        grad = G + self.weight_decay * self._polyak if self.weight_decay > 0 else G.copy()

        step_grad, self._m, self._v = MathematicalPrecision2026.adabound_step(
            grad, self._m, self._v, self.beta1, self.beta2, self.lr, self.final_lr, self.gamma, self.eps, t
        )

        if self.use_muon and step_grad.ndim == 2:
            muon_ortho = MathematicalPrecision2026.newton_schulz5(step_grad, steps=5)
            expected = math.sqrt(min(step_grad.shape))
            self.muon_orthogonality = max(0.0, 1.0 - abs(np.linalg.norm(muon_ortho, ord="fro") - expected) / max(expected, 1.0))
            step_grad = 0.5 * step_grad + 0.5 * muon_ortho

        self._polyak = MathematicalPrecision2026.polyak_step(self._polyak, step_grad, decay=self.polyak_decay)
        self.gsnr_score = MathematicalPrecision2026.gsnr(self._G_sum, self._G_sq_sum, t)

        std_v, mean_v = float(np.std(step_grad)), float(np.mean(np.abs(step_grad))) + 1e-8
        self.adabound_score = max(0.0, min(1.0, 1.0 - (std_v / (mean_v * 3.0))))
        self.boundary_score = max(0.0, min(1.0, float(np.tanh(self.gsnr_score * 0.5))))

        return {
            "adabound_score": self.adabound_score,
            "boundary_score": self.boundary_score,
            "muon_orthogonality": self.muon_orthogonality,
            "gsnr": self.gsnr_score,
        }


# ===========================================================================
# 3. OPTIMIZADOR PRINCIPAL - AdaBoundOptimizer
# ===========================================================================
