class SupervisedLearningNeuralMetrics:
    """Metricas de rendimiento completas del optimizador SLRN 2026."""
    algorithm_name: str
    initial_loss: float
    final_loss: float
    convergence_iterations: int
    bp_momentum_efficiency: float
    sgd_gradient_descent_efficiency: float
    rmsprop_rms_efficiency: float
    adagrad_adaptive_efficiency: float
    adadelta_delta_efficiency: float
    adam_adaptive_momentum: float
    adamax_max_efficiency: float
    amsgrad_maximum_efficiency: float
    adabound_boundary_efficiency: float
    lamb_layer_efficiency: float
    radam_rectified_efficiency: float
    nadam_nesterov_efficiency: float
    novograd_gradient_efficiency: float
    ranger_lookahead_efficiency: float
    supervised_neural_integration_score: float
    overall_score: float
    optimization_time: float
    timestamp: str
    @property
    def loss_reduction_pct(self) -> float:
        """Porcentaje de reduccion de perdida lograda."""
        if self.initial_loss <= 0:
            return 0.0
        return 100.0 * (self.initial_loss - self.final_loss) / self.initial_loss
    def summary(self) -> str:
        """Resumen compacto de una linea."""
        return (
            f"[{self.algorithm_name}] Loss: {self.initial_loss:.4f} -> {self.final_loss:.4f} "
            f"({self.loss_reduction_pct:.1f}% red.) | Score: {self.overall_score:.4f} "
            f"| Iter: {self.convergence_iterations} | {self.timestamp}"
        )
