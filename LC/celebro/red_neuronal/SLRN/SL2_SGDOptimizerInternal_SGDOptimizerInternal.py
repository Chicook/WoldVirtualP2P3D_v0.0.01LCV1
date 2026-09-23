class SGDOptimizerInternal:
    """Motor interno de Stochastic Gradient Descent con EF21 + SGDR + Muon (2026)."""

    def __init__(self, learning_rate: float, momentum: float,
                 dampening: float, weight_decay: float,
                 use_ef21: bool = True, use_muon: bool = True,
                 polyak_decay: float = 0.999, topk_density: float = 0.2):
        self.lr, self.momentum = learning_rate, momentum
        self.dampening, self.weight_decay = dampening, weight_decay
        self.use_ef21, self.use_muon = use_ef21, use_muon
        self.polyak_decay, self.topk_density = polyak_decay, topk_density
        self._velocity: Optional[np.ndarray] = None
        self._error_fb: Optional[np.ndarray] = None
        self._polyak: Optional[np.ndarray] = None
        self._G_sum: Optional[np.ndarray] = None
        self._G_sq: Optional[np.ndarray] = None
        self.step_count = 0
        self.sgd_score = 0.0
        self.gradient_descent_score = 0.0
        self.muon_orthogonality = 0.0
        self.gsnr_score = 0.0
        self.ef21_norm = 0.0
        self.current_lr = learning_rate

    def _init_state(self, G: np.ndarray) -> None:
        self._velocity = np.zeros_like(G)
        self._error_fb = np.zeros_like(G)
        self._polyak = np.zeros_like(G)
        self._G_sum = np.zeros_like(G)
        self._G_sq = np.zeros_like(G)

    def step(self, G: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Ejecuta un paso de SGD 2026 con tecnicas hibridas y devuelve metricas."""
        self.step_count += 1
        t = self.step_count
        if G is None:
            G = np.random.randn(8, 8).astype(np.float64) * 0.1
        if self._velocity is None:
            self._init_state(G)

        self._G_sum += G
        self._G_sq += G ** 2

        self.current_lr = MathematicalPrecision2026.cosine_warm_restart(
            t=t, T_0=20, T_mult=2, eta_min=self.lr * 0.01, eta_max=self.lr
        )
        grad = G + self.weight_decay * self._polyak if self.weight_decay > 0 else G.copy()

        if self.momentum > 0:
            self._velocity = self.momentum * self._velocity + (1.0 - self.dampening) * grad
            step_grad = self._velocity.copy()
        else:
            step_grad = grad

        if self.use_ef21:
            ef_u, self._error_fb = MathematicalPrecision2026.sign_sgd_ef21(step_grad, self._error_fb, self.current_lr)
            self.ef21_norm = float(np.linalg.norm(self._error_fb))

        if self.use_muon and step_grad.ndim == 2:
            muon_ortho = MathematicalPrecision2026.newton_schulz5(step_grad, steps=5)
            expected = math.sqrt(min(step_grad.shape))
            self.muon_orthogonality = max(0.0, 1.0 - abs(np.linalg.norm(muon_ortho, ord="fro") - expected) / max(expected, 1.0))
            step_grad = 0.5 * step_grad + 0.5 * muon_ortho

        step_grad = MathematicalPrecision2026.top_k_sparsify(step_grad, density=self.topk_density)
        self._polyak = MathematicalPrecision2026.polyak_step(self._polyak, step_grad, decay=self.polyak_decay)
        self.gsnr_score = MathematicalPrecision2026.gsnr(self._G_sum, self._G_sq, t)

        std_v, mean_v = float(np.std(step_grad)), float(np.mean(np.abs(step_grad))) + 1e-8
        self.sgd_score = max(0.0, min(1.0, 1.0 - (std_v / (mean_v * 3.0))))
        self.gradient_descent_score = max(0.0, min(1.0, float(np.tanh(self.gsnr_score * 0.5))))

        return {
            "sgd_score": self.sgd_score,
            "gradient_descent_score": self.gradient_descent_score,
            "muon_orthogonality": self.muon_orthogonality,
            "gsnr": self.gsnr_score,
            "effective_lr": self.current_lr,
            "ef21_norm": self.ef21_norm,
        }


# ===========================================================================
# 3. OPTIMIZADOR PRINCIPAL - SGDOptimizer
# ===========================================================================
