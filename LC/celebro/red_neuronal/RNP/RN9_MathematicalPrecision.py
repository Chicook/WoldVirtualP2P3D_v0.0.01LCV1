class MathematicalPrecision:
    @staticmethod
    def arrays(values: List[np.ndarray]) -> List[np.ndarray]:
        return [np.asarray(v, dtype=float) for v in values]
    @staticmethod
    def norm(values: List[np.ndarray]) -> float:
        return float(np.sqrt(sum(float(np.sum(v * v)) for v in values)))
    @staticmethod
    def clip(gradients: List[np.ndarray], maximum: float) -> List[np.ndarray]:
        values = MathematicalPrecision.arrays(gradients); total = MathematicalPrecision.norm(values)
        return [v * (maximum / total) for v in values] if maximum > 0 and total > maximum else values
    @staticmethod
    def normalize(gradients: List[np.ndarray], epsilon: float) -> List[np.ndarray]:
        values = MathematicalPrecision.arrays(gradients); total = MathematicalPrecision.norm(values)
        return [v / max(total, epsilon) for v in values] if total > epsilon else values
    @staticmethod
    def signal_ratio(gradients: List[np.ndarray], epsilon: float) -> float:
        values = MathematicalPrecision.arrays(gradients)
        mean = float(np.mean([float(np.mean(np.abs(v))) for v in values])) if values else 0.0
        dev = float(np.mean([float(np.std(v)) for v in values])) if values else 0.0
        return float(mean / max(dev, epsilon))
    @staticmethod
    def condition(gradients: List[np.ndarray], epsilon: float) -> float:
        norms = [v for v in [float(np.linalg.norm(v)) for v in MathematicalPrecision.arrays(gradients)] if v > epsilon]
        return float(max(norms) / max(min(norms), epsilon)) if norms else 1.0
    @staticmethod
    def cosine(first: List[np.ndarray], second: List[np.ndarray], epsilon: float) -> float:
        a, b = MathematicalPrecision.arrays(first), MathematicalPrecision.arrays(second)
        if len(a) != len(b) or not a: return 0.0
        dot = sum(float(np.sum(x * y)) for x, y in zip(a, b))
        return float(max(-1.0, min(1.0, dot / max(MathematicalPrecision.norm(a) * MathematicalPrecision.norm(b), epsilon))))
    @staticmethod
    def newton_schulz_orthogonalization(mat: np.ndarray, steps: int = 5, epsilon: float = 1e-8) -> np.ndarray:
        """Muon (2025/2026): aproximación ortogonal rápida mediante polinomios Newton-Schulz grado 5."""
        if mat.ndim < 2:
            return mat / max(float(np.linalg.norm(mat)), epsilon)
        orig_shape = mat.shape
        x = mat.reshape(orig_shape[0], -1) if mat.ndim > 2 else mat.copy()
        transposed = x.shape[0] < x.shape[1]
        if transposed: x = x.T
        norm = np.linalg.norm(x)
        if norm > epsilon: x = x / norm
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(steps):
            a_mat = x.T @ x
            b_mat = b * a_mat + c * (a_mat @ a_mat)
            x = x @ (a * np.eye(a_mat.shape[0]) + b_mat)
        if transposed: x = x.T
        return x.reshape(orig_shape)
    @staticmethod
    def update(weights: List[np.ndarray], gradients: List[np.ndarray],
               momentum: List[np.ndarray], exp_avg: List[np.ndarray],
               exp_avg_sq: List[np.ndarray], sgd_momentum: List[np.ndarray],
               step: int, learning_rate: float, beta1: float, beta2: float,
               epsilon: float, weight_decay: float, switch_iter: int,
               switch_threshold: float, gradient_clipping: float,
               normalize_gradients: bool, decoupled_decay: bool, nesterov: bool,
               trust_ratio: bool, polyak_decay: float, update_clip: float,
               precision_epsilon: float, polyak_weights: List[np.ndarray],
               muon_ns_steps: int = 5, muon_orthogonal: bool = True) -> tuple:
        old_weights = MathematicalPrecision.arrays(weights)
        gradients = MathematicalPrecision.arrays(gradients)
        clipped = MathematicalPrecision.clip(gradients, gradient_clipping)
        gradients = MathematicalPrecision.normalize(clipped, precision_epsilon) if normalize_gradients else clipped
        momentum = momentum or [np.zeros_like(v) for v in old_weights]
        exp_avg = exp_avg or [np.zeros_like(v) for v in old_weights]
        exp_avg_sq = exp_avg_sq or [np.zeros_like(v) for v in old_weights]
        sgd_momentum = sgd_momentum or [np.zeros_like(v) for v in old_weights]
        weight_norm, gradient_norm = MathematicalPrecision.norm(old_weights), MathematicalPrecision.norm(gradients)
        new_momentum, new_exp_avg, new_exp_avg_sq, new_sgd, directions, alignments = [], [], [], [], [], []
        ortho_scores = []
        for weight, gradient, first, second, sq, sgd in zip(old_weights, gradients, momentum, exp_avg, exp_avg_sq, sgd_momentum):
            first_new = beta1 * first + (1.0 - beta1) * gradient
            second_new = beta2 * second + (1.0 - beta2) * np.square(gradient)
            first_hat = first_new / max(1.0 - beta1 ** step, precision_epsilon)
            second_hat = np.maximum(second_new / max(1.0 - beta2 ** step, precision_epsilon), 0.0)
            adaptive = first_hat / (np.sqrt(second_hat) + epsilon)
            alignment = MathematicalPrecision.cosine([adaptive], [gradient], precision_epsilon)
            phase = 'sgd' if step >= switch_iter or alignment >= switch_threshold else 'adam'
            if phase == 'sgd':
                base_dir = gradient + (sgd if nesterov else np.zeros_like(gradient))
                direction = MathematicalPrecision.newton_schulz_orthogonalization(base_dir, muon_ns_steps, precision_epsilon) if muon_orthogonal else base_dir
            else:
                direction = adaptive
            new_momentum.append(first_new); new_exp_avg.append(first_new)
            new_exp_avg_sq.append(second_new); new_sgd.append(direction)
            directions.append(direction); alignments.append(alignment)
            ortho_scores.append(float(np.mean(np.abs(direction))))
        alignment = float(np.mean(alignments)) if alignments else 0.0
        phase = 'sgd' if step >= switch_iter or alignment >= switch_threshold else 'adam'
        ratios = [weight_norm / max(float(np.linalg.norm(g)), precision_epsilon) for g in gradients] if trust_ratio else [1.0]
        ratio = float(np.clip(np.mean(ratios), 0.1, 10.0)) if ratios else 1.0
        updates = [-learning_rate * ratio * d for d in directions]
        if decoupled_decay:
            updates = [u - learning_rate * weight_decay * w for u, w in zip(updates, old_weights)]
        update_norm = MathematicalPrecision.norm(updates)
        if update_clip > 0 and update_norm > update_clip:
            updates = [u * (update_clip / update_norm) for u in updates]
        new_weights = [w + u for w, u in zip(old_weights, updates)]
        polyak_weights = polyak_weights or [np.zeros_like(v) for v in old_weights]
        polyak_weights = [polyak_decay * p + (1.0 - polyak_decay) * w for p, w in zip(polyak_weights, new_weights)]
        condition = MathematicalPrecision.condition(gradients, precision_epsilon)
        signal = MathematicalPrecision.signal_ratio(gradients, precision_epsilon)
        similarity = MathematicalPrecision.cosine(old_weights, updates, precision_epsilon)
        efficiency = 1.0 / (1.0 + update_norm / max(gradient_norm, precision_epsilon))
        distance = MathematicalPrecision.norm([a - b for a, b in zip(polyak_weights, new_weights)])
        polyak_score = 1.0 / (1.0 + distance / max(weight_norm, precision_epsilon))
        precision = (1.0 / (1.0 + condition / 1000.0)) * (0.5 + 0.5 * max(0.0, similarity))
        stats = {'update_norm': update_norm, 'gradient_norm': gradient_norm, 'weight_norm': weight_norm,
                 'condition_estimate': condition, 'gradient_signal_ratio': signal,
                 'cosine_similarity': similarity, 'update_efficiency': efficiency,
                 'switch_iteration': step if phase == 'sgd' else 0, 'phase': phase,
                 'momentum_alignment': alignment, 'polyak_score': polyak_score,
                 'trust_ratio_mean': ratio, 'precision_score': precision,
                 'swats_efficiency': efficiency * max(0.0, alignment),
                 'muon_orthogonality': float(np.mean(ortho_scores)) if ortho_scores else 1.0}
        return new_weights, new_momentum, new_exp_avg, new_exp_avg_sq, new_sgd, stats, polyak_weights

