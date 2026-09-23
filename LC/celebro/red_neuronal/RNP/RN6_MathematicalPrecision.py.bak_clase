class MathematicalPrecision:
    @staticmethod
    def arrays(values: List[np.ndarray]) -> List[np.ndarray]:
        return [np.asarray(v, dtype=float) for v in values]

    @staticmethod
    def norm(values: List[np.ndarray]) -> float:
        return float(np.sqrt(sum(float(np.sum(v * v)) for v in values)))

    @staticmethod
    def clip(gradients: List[np.ndarray], maximum: float) -> List[np.ndarray]:
        values = MathematicalPrecision.arrays(gradients)
        total = MathematicalPrecision.norm(values)
        return [v * (maximum / total) for v in values] if maximum > 0 and total > maximum else values

    @staticmethod
    def signal_ratio(gradients: List[np.ndarray], epsilon: float) -> float:
        values = MathematicalPrecision.arrays(gradients)
        mean = float(np.mean([float(np.mean(np.abs(v))) for v in values])) if values else 0.0
        deviation = float(np.mean([float(np.std(v)) for v in values])) if values else 0.0
        return float(mean / max(deviation, epsilon))

    @staticmethod
    def condition(gradients: List[np.ndarray], epsilon: float) -> float:
        norms = [float(np.linalg.norm(v)) for v in MathematicalPrecision.arrays(gradients)]
        norms = [v for v in norms if v > epsilon]
        return float(max(norms) / max(min(norms), epsilon)) if norms else 1.0

    @staticmethod
    def cosine(first: List[np.ndarray], second: List[np.ndarray], epsilon: float) -> float:
        a, b = MathematicalPrecision.arrays(first), MathematicalPrecision.arrays(second)
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(float(np.sum(x * y)) for x, y in zip(a, b))
        return float(max(-1.0, min(1.0, dot / max(MathematicalPrecision.norm(a) * MathematicalPrecision.norm(b), epsilon))))

    @staticmethod
    def update(weights: List[np.ndarray], gradients: List[np.ndarray],
               momentum: List[np.ndarray], variance: List[np.ndarray], step: int,
               learning_rate: float, beta1: float, beta2: float, epsilon: float,
               weight_decay: float, ams_bound: bool, gradient_clipping: float,
               update_clip: float, precision_epsilon: float) -> tuple:
        old_weights = MathematicalPrecision.arrays(weights)
        old_gradients = MathematicalPrecision.clip(gradients, gradient_clipping)
        momentum = momentum or [np.zeros_like(v) for v in old_weights]
        variance = variance or [np.zeros_like(v) for v in old_weights]
        new_momentum, new_variance, updates = [], [], []
        for weight, gradient, first, second in zip(old_weights, old_gradients, momentum, variance):
            first_new = beta1 * first + (1.0 - beta1) * gradient
            second_raw = beta2 * second + (1.0 - beta2) * np.square(gradient - first_new)
            second_new = np.maximum(second_raw, second) if ams_bound else second_raw
            first_hat = first_new / max(1.0 - beta1 ** step, precision_epsilon)
            second_hat = second_new / max(1.0 - beta2 ** step, precision_epsilon)
            update = -learning_rate * (first_hat / (np.sqrt(second_hat) + epsilon) + weight_decay * weight)
            new_momentum.append(first_new); new_variance.append(second_new); updates.append(update)
        update_norm = MathematicalPrecision.norm(updates)
        if update_clip > 0 and update_norm > update_clip:
            updates = [v * (update_clip / update_norm) for v in updates]
        gradient_norm, weight_norm = MathematicalPrecision.norm(old_gradients), MathematicalPrecision.norm(old_weights)
        condition = MathematicalPrecision.condition(old_gradients, precision_epsilon)
        signal = MathematicalPrecision.signal_ratio(old_gradients, precision_epsilon)
        similarity = MathematicalPrecision.cosine(old_weights, updates, precision_epsilon)
        efficiency = 1.0 / (1.0 + update_norm / max(gradient_norm, precision_epsilon))
        stats = {'update_norm': update_norm, 'gradient_norm': gradient_norm, 'weight_norm': weight_norm,
                 'condition_estimate': condition, 'gradient_signal_ratio': signal,
                 'cosine_similarity': similarity, 'update_efficiency': efficiency,
                 'precision_score': 1.0 / (1.0 + condition / 1000.0)}
        return [w + u for w, u in zip(old_weights, updates)], new_momentum, new_variance, updates, stats


