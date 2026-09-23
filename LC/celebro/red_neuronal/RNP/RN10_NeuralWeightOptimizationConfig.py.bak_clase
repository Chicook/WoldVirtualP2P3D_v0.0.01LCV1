class NeuralWeightOptimizationConfig:
    """Configuración unificada para la optimización integrada de pesos neuronales 2026."""

    def __init__(self, learning_rate: float = 0.001, weight_decay: float = 0.01,
                 max_iterations: int = 1000, gradient_clipping: float = 1.0,
                 normalize_gradients: bool = True, decoupled_decay: bool = True,
                 nesterov: bool = True, trust_ratio: bool = True,
                 polyak_decay: float = 0.999, update_clip: float = 10.0,
                 precision_epsilon: float = 1e-12, muon_ns_steps: int = 5,
                 muon_orthogonal: bool = True, sam_rho: float = 0.05,
                 sam_adaptive: bool = True, kan_spline_order: int = 3,
                 kan_grid_size: int = 5, hybrid_strategy: str = "adaptive_ensemble"):
        self.learning_rate = max(0.0, float(learning_rate))
        self.weight_decay = max(0.0, float(weight_decay))
        self.max_iterations = max(1, int(max_iterations))
        self.gradient_clipping = max(0.0, float(gradient_clipping))
        self.normalize_gradients, self.decoupled_decay = bool(normalize_gradients), bool(decoupled_decay)
        self.nesterov, self.trust_ratio = bool(nesterov), bool(trust_ratio)
        self.polyak_decay = max(0.0, min(1.0, float(polyak_decay)))
        self.update_clip = max(0.0, float(update_clip))
        self.precision_epsilon = max(float(precision_epsilon), 1e-15)
        self.muon_ns_steps, self.muon_orthogonal = max(1, int(muon_ns_steps)), bool(muon_orthogonal)
        self.sam_rho, self.sam_adaptive = max(0.0, float(sam_rho)), bool(sam_adaptive)
        self.kan_spline_order = max(1, int(kan_spline_order))
        self.kan_grid_size = max(3, int(kan_grid_size))
        self.hybrid_strategy = str(hybrid_strategy)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


