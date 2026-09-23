class AdaGradOptimizerInternal:
    """Motor interno de AdaGrad con AdaGrad-Norm + Muon + GSNR (2026)."""

    def __init__(self, learning_rate: float, eps: float,
                 weight_decay: float, use_norm_variant: bool = True,
                 use_muon: bool = True, polyak_decay: float = 0.999,
                 acc_decay: float = 0.9995):
        self.lr = learning_rate
        self.eps = eps
        self.weight_decay = weight_decay
        self.use_norm_variant = use_norm_variant
        self.use_muon = use_muon
        self.polyak_decay = polyak_decay
        self.acc_decay = acc_decay
        self._G_acc: Optional[np.ndarray] = None
        self._norm_b: float = 0.0
        self._polyak: Optional[np.ndarray] = None
        self._G_sum: Optional[np.ndarray] = None
        self._G_sq_sum: Optional[np.ndarray] = None
        self.step_count = 0
        self.adagrad_score = 0.0
        self.adaptive_score = 0.0
        self.muon_orthogonality = 0.0
        self.gsnr_score = 0.0

    def _init_state(self, G: np.ndarray) -> None:
        self._G_acc = np.zeros_like(G)
        self._polyak = np.zeros_like(G)
        self._G_sum = np.zeros_like(G)
        self._G_sq_sum = np.zeros_like(G)
        self._norm_b = 0.0

    def step(self, G: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Ejecuta un paso de optimizacion AdaGrad 2026 y devuelve metricas."""
        self.step_count += 1
        t = self.step_count
        if G is None:
            G = np.random.randn(8, 8).astype(np.float64) * 0.1
        if self._G_acc is None:
            self._init_state(G)

        self._G_sum += G
        self._G_sq_sum += G ** 2

        grad = G + self.weight_decay * self._polyak if self.weight_decay > 0 else G.copy()

        # Actualizacion acumulativa controlada
        self._G_acc = MathematicalPrecision2026.decayed_accumulator(self._G_acc, grad, decay=self.acc_decay)
        diag_step = grad / (np.sqrt(self._G_acc) + self.eps)

        if self.use_norm_variant:
            norm_step, self._norm_b = MathematicalPrecision2026.adagrad_norm_step(
                grad, self._norm_b, self.lr, self.eps
            )
            step_grad = 0.5 * (self.lr * diag_step) + 0.5 * norm_step
        else:
            step_grad = self.lr * diag_step

        if self.use_muon and step_grad.ndim == 2:
            muon_ortho = MathematicalPrecision2026.newton_schulz5(step_grad, steps=5)
            expected = math.sqrt(min(step_grad.shape))
            self.muon_orthogonality = max(0.0, 1.0 - abs(np.linalg.norm(muon_ortho, ord="fro") - expected) / max(expected, 1.0))
            step_grad = 0.5 * step_grad + 0.5 * muon_ortho

        self._polyak = MathematicalPrecision2026.polyak_step(self._polyak, step_grad, decay=self.polyak_decay)
        self.gsnr_score = MathematicalPrecision2026.gsnr(self._G_sum, self._G_sq_sum, t)

        std_v, mean_v = float(np.std(step_grad)), float(np.mean(np.abs(step_grad))) + 1e-8
        self.adagrad_score = max(0.0, min(1.0, 1.0 - (std_v / (mean_v * 3.0))))
        self.adaptive_score = max(0.0, min(1.0, float(np.tanh(self.gsnr_score * 0.5))))

        return {
            "adagrad_score": self.adagrad_score,
            "adaptive_score": self.adaptive_score,
            "muon_orthogonality": self.muon_orthogonality,
            "gsnr": self.gsnr_score,
            "accumulator_norm": float(np.linalg.norm(self._G_acc)),
        }


# ===========================================================================
# 3. OPTIMIZADOR PRINCIPAL - AdaGradOptimizer
# ===========================================================================
