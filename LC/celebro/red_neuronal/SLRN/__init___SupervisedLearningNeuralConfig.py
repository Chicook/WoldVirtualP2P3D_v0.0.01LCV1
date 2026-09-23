class SupervisedLearningNeuralConfig:
    """Configuracion unificada para todos los optimizadores de SLRN 2026."""
    # --- Generales ---
    learning_rate: float = 0.001
    max_iterations: int = 1000
    batch_size: int = 32
    weight_decay: float = 0.0001
    random_state: int = 42
    # --- Backpropagation / Momentum ---
    bp_momentum: float = 0.9
    bp_nesterov: bool = True
    # --- SGD ---
    sgd_momentum: float = 0.9
    sgd_dampening: float = 0.0
    # --- RMSprop ---
    rmsprop_alpha: float = 0.99
    rmsprop_eps: float = 1e-8
    # --- AdaGrad ---
    adagrad_eps: float = 1e-10
    # --- AdaDelta ---
    adadelta_rho: float = 0.9
    adadelta_eps: float = 1e-6
    # --- Adam / AdamW ---
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    # --- Adamax ---
    adamax_beta1: float = 0.9
    adamax_beta2: float = 0.999
    adamax_eps: float = 1e-8
    # --- AMSGrad ---
    amsgrad_beta1: float = 0.9
    amsgrad_beta2: float = 0.999
    amsgrad_eps: float = 1e-8
    # --- AdaBound ---
    adabound_final_lr: float = 0.1
    adabound_gamma: float = 0.001
    # --- LAMB / RAdam / NAdam / NovoGrad / Ranger (reserved) ---
    lamb_beta1: float = 0.9
    lamb_beta2: float = 0.999
    lamb_eps: float = 1e-8
    radam_beta1: float = 0.9
    radam_beta2: float = 0.999
    radam_eps: float = 1e-8
    nadam_beta1: float = 0.9
    nadam_beta2: float = 0.999
    nadam_eps: float = 1e-8
    nadam_momentum_decay: float = 0.004
    novograd_beta1: float = 0.9
    novograd_beta2: float = 0.999
    novograd_eps: float = 1e-8
    ranger_beta1: float = 0.9
    ranger_beta2: float = 0.999
    ranger_eps: float = 1e-8
    ranger_lookahead_k: int = 5
    ranger_lookahead_alpha: float = 0.5
    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
