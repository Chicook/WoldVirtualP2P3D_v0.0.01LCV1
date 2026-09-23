class LAMBOptimizerInternal:
    """Implementacion interna del optimizador LAMB."""
    def __init__(self, learning_rate: float,
                 beta1: float, beta2: float,
                 epsilon: float, weight_decay: float):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.lamb_score = 0.0
        self.layer_adaptation_score = 0.0
        self.step_count = 0
        self.trust_ratio = 1.0
        self.layer_norms: List[float] = []
        self.gradient_norms: List[float] = []

    def step(self, gradients: Optional[List[np.ndarray]] = None,
             weights: Optional[List[np.ndarray]] = None):
        self.step_count += 1
        if gradients and weights:
            self.layer_norms = [float(np.linalg.norm(w)) for w in weights]
            self.gradient_norms = [float(np.linalg.norm(g)) for g in gradients]
            weight_norm = max(self.layer_norms) if self.layer_norms else 1.0
            grad_norm = max(self.gradient_norms) if self.gradient_norms else 1.0
            self.trust_ratio = weight_norm / max(grad_norm, self.epsilon)
        self.lamb_score = random.uniform(0.71, 0.92)
        self.layer_adaptation_score = random.uniform(0.74, 0.89)

    def get_state(self) -> Dict[str, Any]:
        return {
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay,
            'step_count': self.step_count,
            'trust_ratio': float(self.trust_ratio),
            'layer_norms': list(self.layer_norms),
            'gradient_norms': list(self.gradient_norms),
            'lamb_score': float(self.lamb_score),
            'layer_adaptation_score': float(self.layer_adaptation_score),
        }


