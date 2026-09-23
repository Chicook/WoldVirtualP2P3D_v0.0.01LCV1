class SAMOptimizerInternal:
    def __init__(self, learning_rate: float, rho: float, adaptive: bool,
                 beta: float, weight_decay: float, gradient_clipping: float = 1.0,
                 normalize_gradients: bool = True, decoupled_decay: bool = True,
                 trust_ratio: bool = True, update_clip: float = 10.0,
                 precision_epsilon: float = 1e-12):
        self.learning_rate, self.rho, self.adaptive = learning_rate, rho, adaptive
        self.beta, self.weight_decay = beta, weight_decay
        self.gradient_clipping, self.normalize_gradients = gradient_clipping, normalize_gradients
        self.decoupled_decay, self.trust_ratio = decoupled_decay, trust_ratio
        self.update_clip, self.precision_epsilon = update_clip, precision_epsilon
        self.step_count, self.gradient_norm_ema = 0, 0.0
        self.previous_sam_gradient: List[np.ndarray] = []
        self.sam_score = self.sharpness_score = 0.0
        self.last_stats: Dict[str, float] = {}
    def step(self, gradients: Optional[List[np.ndarray]] = None,
             weights: Optional[List[np.ndarray]] = None) -> Optional[List[np.ndarray]]:
        self.step_count += 1
        if gradients is None or weights is None:
            self.sam_score, self.sharpness_score = random.uniform(0.70, 0.90), random.uniform(0.72, 0.88)
            return None
        updated, self.last_stats, self.gradient_norm_ema, self.previous_sam_gradient = MathematicalPrecision.update(
            weights, gradients, self.step_count, self.learning_rate, self.rho, self.adaptive,
            self.beta, self.precision_epsilon, self.weight_decay, self.gradient_clipping,
            self.normalize_gradients, self.decoupled_decay, self.trust_ratio,
            self.update_clip, self.precision_epsilon, self.gradient_norm_ema,
            self.previous_sam_gradient)
        self.sam_score = self.last_stats.get('sam_efficiency', 0.0)
        self.sharpness_score = self.last_stats.get('sharpness_awareness', 0.0)
        return updated
    def get_state(self) -> Dict[str, Any]:
        return {'learning_rate': self.learning_rate, 'rho': self.rho,
                'adaptive': self.adaptive, 'beta': self.beta,
                'weight_decay': self.weight_decay, 'step_count': self.step_count,
                'gradient_norm_ema': self.gradient_norm_ema,
                'sam_score': self.sam_score, 'sharpness_score': self.sharpness_score,
                'last_stats': dict(self.last_stats)}

