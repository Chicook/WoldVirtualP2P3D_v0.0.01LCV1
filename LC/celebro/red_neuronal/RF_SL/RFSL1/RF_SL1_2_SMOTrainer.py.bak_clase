class SMOTrainer:
    def __init__(self, C: float = 1.0, tol: float = 1e-3, max_iter: int = 1000):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self._objective_history: List[float] = []

    def optimize(self, K: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        n = len(y)
        alphas = np.zeros(n)
        bias = 0.0
        for _ in range(self.max_iter):
            alpha_prev = alphas.copy()
            for i in range(n):
                Ei = np.sum(alphas * y * K[:, i]) + bias - y[i]
                if (y[i] * Ei < -self.tol and alphas[i] < self.C) or (y[i] * Ei > self.tol and alphas[i] > 0):
                    j = np.random.randint(0, n)
                    Ej = np.sum(alphas * y * K[:, j]) + bias - y[j]
                    alpha_i_old, alpha_j_old = alphas[i], alphas[j]
                    if y[i] != y[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[i] + alphas[j] - self.C)
                        H = min(self.C, alphas[i] + alphas[j])
                    if L == H: continue
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0: continue
                    alphas[j] -= y[j] * (Ei - Ej) / eta
                    alphas[j] = np.clip(alphas[j], L, H)
                    if abs(alphas[j] - alpha_j_old) < 1e-5: continue
                    alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])
            if np.max(np.abs(alphas - alpha_prev)) < self.tol: break
        sv = alphas > 1e-5
        self._objective_history.append(float(np.sum(alphas[sv])))
        return alphas, bias

    def get_objective_history(self) -> List[float]: return self._objective_history.copy()

    def is_converged(self, tol: float = 1e-4) -> bool:
        if len(self._objective_history) < 2: return False
        return abs(self._objective_history[-1] - self._objective_history[-2]) < tol

    def reset(self) -> None: self._objective_history.clear()


