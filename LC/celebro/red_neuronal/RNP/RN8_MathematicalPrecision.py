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
    def update(weights: List[np.ndarray], gradients: List[np.ndarray], step: int,
               learning_rate: float, rho: float, adaptive: bool, beta: float,
               epsilon: float, weight_decay: float, gradient_clipping: float,
               normalize_gradients: bool, decoupled_decay: bool, trust_ratio: bool,
               update_clip: float, precision_epsilon: float,
               gradient_norm_ema: float, previous_sam_gradient: List[np.ndarray]) -> tuple:
        old_weights = MathematicalPrecision.arrays(weights)
        raw_gradients = MathematicalPrecision.arrays(gradients)
        clipped = MathematicalPrecision.clip(raw_gradients, gradient_clipping)
        gradients = MathematicalPrecision.normalize(clipped, precision_epsilon) if normalize_gradients else clipped
        weight_norm, gradient_norm = MathematicalPrecision.norm(old_weights), MathematicalPrecision.norm(gradients)
        gradient_norm_ema = beta * gradient_norm_ema + (1.0 - beta) * gradient_norm ** 2
        adaptive_scale = np.sqrt(max(gradient_norm_ema, precision_epsilon)) / max(gradient_norm, precision_epsilon) if adaptive else 1.0
        perturbations = [rho * adaptive_scale * g / max(float(np.linalg.norm(g)), precision_epsilon) for g in gradients]
        perturbation_norm = MathematicalPrecision.norm(perturbations)
        curvature = rho * gradient_norm / max(weight_norm, precision_epsilon)
        perturbed_gradients = [g + curvature * p for g, p in zip(gradients, perturbations)]
        sam_gradient = [(g + pg) * 0.5 for g, pg in zip(gradients, perturbed_gradients)]
        if previous_sam_gradient:
            alignment = abs(MathematicalPrecision.cosine(previous_sam_gradient, sam_gradient, precision_epsilon))
        else:
            alignment = 1.0
        ratios = [weight_norm / max(float(np.linalg.norm(g)), precision_epsilon) for g in gradients] if trust_ratio else [1.0]
        ratio = float(np.clip(np.mean(ratios), 0.1, 10.0)) if ratios else 1.0
        updates = [-learning_rate * ratio * g for g in sam_gradient]
        if decoupled_decay:
            updates = [u - learning_rate * weight_decay * w for u, w in zip(updates, old_weights)]
        update_norm = MathematicalPrecision.norm(updates)
        if update_clip > 0 and update_norm > update_clip:
            updates = [u * (update_clip / update_norm) for u in updates]
        condition = MathematicalPrecision.condition(gradients, precision_epsilon)
        signal = MathematicalPrecision.signal_ratio(gradients, precision_epsilon)
        similarity = MathematicalPrecision.cosine(old_weights, updates, precision_epsilon)
        efficiency = 1.0 / (1.0 + update_norm / max(gradient_norm, precision_epsilon))
        flatness = 1.0 / (1.0 + perturbation_norm / max(gradient_norm, precision_epsilon))
        sharpness = 1.0 / (1.0 + curvature)
        precision = (1.0 / (1.0 + condition / 1000.0)) * (0.5 + 0.5 * max(0.0, similarity))
        stats = {'update_norm': update_norm, 'gradient_norm': gradient_norm, 'weight_norm': weight_norm,
                 'condition_estimate': condition, 'gradient_signal_ratio': signal,
                 'cosine_similarity': similarity, 'update_efficiency': efficiency,
                 'perturbation_norm': perturbation_norm, 'curvature_estimate': curvature,
                 'flatness_score': flatness, 'sharpness_awareness': sharpness,
                 'trust_ratio_mean': ratio, 'precision_score': precision,
                 'sam_efficiency': efficiency * flatness, 'momentum_alignment': alignment}
        return [w + u for w, u in zip(old_weights, updates)], stats, gradient_norm_ema, sam_gradient

