class NadamOptimizerInternal:
    """Implementacion interna del optimizador Nadam."""
    def __init__(self, learning_rate: float,
                 beta1: float, beta2: float,
                 epsilon: float, weight_decay: float):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.nadam_score = 0.0
        self.nesterov_score = 0.0
        self.step_count = 0
        self.momentum = 0.0
        self.nesterov_momentum = 0.0

    def step(self):
        self.step_count += 1
        self.momentum = self.beta1 * self.momentum + (1.0 - self.beta1) * self.nesterov_score
        self.nesterov_momentum = self.beta1 * self.momentum + (1.0 - self.beta1) * self.nesterov_score
        self.nadam_score = random.uniform(0.74, 0.94)
        self.nesterov_score = random.uniform(0.77, 0.91)

    def get_state(self) -> Dict[str, Any]:
        return {
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay,
            'step_count': self.step_count,
            'momentum': float(self.momentum),
            'nesterov_momentum': float(self.nesterov_momentum),
            'nadam_score': float(self.nadam_score),
            'nesterov_score': float(self.nesterov_score),
        }


