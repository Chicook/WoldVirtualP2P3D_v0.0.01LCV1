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
    def normalize(gradients: List[np.ndarray], epsilon: float) -> List[np.ndarray]:
        values = MathematicalPrecision.arrays(gradients)
        total = MathematicalPrecision.norm(values)
        return [v / max(total, epsilon) for v in values] if total > epsilon else values
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
    def sparsity(updates: List[np.ndarray]) -> float:
        values = MathematicalPrecision.arrays(updates)
        return float(np.mean([float(np.mean(v == 0)) for v in values])) if values else 0.0
    @staticmethod
    def update(weights: List[np.ndarray], gradients: List[np.ndarray], momentum: List[np.ndarray],
               step: int, learning_rate: float, beta1: float, beta2: float,
               epsilon: float, weight_decay: float, gradient_clipping: float,
               normalize_gradients: bool, decoupled_decay: bool, trust_ratio: bool,
               sharpness_rho: float, update_clip: float, precision_epsilon: float) -> tuple:
        old_weights = MathematicalPrecision.arrays(weights)
        raw_gradients = MathematicalPrecision.arrays(gradients)
        clipped = MathematicalPrecision.clip(raw_gradients, gradient_clipping)
        gradients = MathematicalPrecision.normalize(clipped, precision_epsilon) if normalize_gradients else clipped
        momentum = momentum or [np.zeros_like(v) for v in old_weights]
        weight_norm, gradient_norm = MathematicalPrecision.norm(old_weights), MathematicalPrecision.norm(gradients)
        sharpness = sharpness_rho * gradient_norm / max(weight_norm, precision_epsilon)
        adjusted = [g * (1.0 + sharpness) for g in gradients]
        direction = [np.sign(beta1 * m - (1.0 - beta1) * g) for m, g in zip(momentum, adjusted)]
        new_momentum = [beta2 * m - (1.0 - beta2) * g for m, g in zip(momentum, adjusted)]
        ratios = [weight_norm / max(float(np.linalg.norm(g)), precision_epsilon) for g in gradients] if trust_ratio else [1.0]
        ratio = float(np.clip(np.mean(ratios), 0.1, 10.0)) if ratios else 1.0
        updates = [-learning_rate * ratio * d for d in direction]
        if decoupled_decay:
            updates = [u - learning_rate * weight_decay * w for u, w in zip(updates, old_weights)]
        update_norm = MathematicalPrecision.norm(updates)
        if update_clip > 0 and update_norm > update_clip:
            updates = [u * (update_clip / update_norm) for u in updates]
        condition = MathematicalPrecision.condition(gradients, precision_epsilon)
        signal = MathematicalPrecision.signal_ratio(gradients, precision_epsilon)
        similarity = MathematicalPrecision.cosine(old_weights, updates, precision_epsilon)
        efficiency = 1.0 / (1.0 + update_norm / max(gradient_norm, precision_epsilon))
        sparse = MathematicalPrecision.sparsity(updates)
        precision = (1.0 / (1.0 + condition / 1000.0)) * (0.5 + 0.5 * max(0.0, similarity))
        stats = {'update_norm': update_norm, 'gradient_norm': gradient_norm, 'weight_norm': weight_norm,
                 'condition_estimate': condition, 'gradient_signal_ratio': signal,
                 'cosine_similarity': similarity, 'update_efficiency': efficiency,
                 'sparsity_ratio': sparse, 'sharpness_score': 1.0 / (1.0 + sharpness),
                 'trust_ratio_mean': ratio, 'precision_score': precision,
                 'sharpness': sharpness, 'momentum_alignment': abs(similarity)}
        return [w + u for w, u in zip(old_weights, updates)], new_momentum, updates, stats

