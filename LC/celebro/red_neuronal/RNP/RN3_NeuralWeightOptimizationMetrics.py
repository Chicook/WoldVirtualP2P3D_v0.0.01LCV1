class NeuralWeightOptimizationMetrics:
    def __init__(self, algorithm_name: str = "Lookahead",
                 initial_loss: float = 0.0, final_loss: float = 0.0,
                 convergence_iterations: int = 0,
                 adamw_weight_decay_efficiency: float = 0.0,
                 radam_rectification_stability: float = 0.0,
                 lookahead_convergence_speed: float = 0.0,
                 lookahead_efficiency: float = 0.0,
                 lookahead_integration_score: float = 0.0,
                 lookahead_stability: float = 0.0,
                 lookahead_trend: str = "stable",
                 overall_score: float = 0.0,
                 optimization_time: float = 0.0,
                 timestamp: str = ""):
        self.algorithm_name = algorithm_name
        self.initial_loss = initial_loss
        self.final_loss = final_loss
        self.convergence_iterations = convergence_iterations
        self.adamw_weight_decay_efficiency = adamw_weight_decay_efficiency
        self.radam_rectification_stability = radam_rectification_stability
        self.lookahead_convergence_speed = lookahead_convergence_speed
        self.lookahead_efficiency = lookahead_efficiency
        self.lookahead_integration_score = lookahead_integration_score
        self.lookahead_stability = lookahead_stability
        self.lookahead_trend = lookahead_trend
        self.overall_score = overall_score
        self.optimization_time = optimization_time
        self.timestamp = timestamp


