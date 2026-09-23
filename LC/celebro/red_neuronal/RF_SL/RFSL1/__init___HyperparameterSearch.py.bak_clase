class HyperparameterSearch:
    def __init__(self, param_grid: Dict[str, List[Any]]):
        self.param_grid = param_grid
        self._results: List[Dict[str, Any]] = []

    def enumerate_combinations(self) -> List[Dict[str, Any]]:
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combos = []
        for combo in itertools.product(*values):
            combos.append(dict(zip(keys, combo)))
        return combos

    def search(self, X: np.ndarray, y: np.ndarray,
                  model_type: str, metric: str = "r2") -> Dict[str, Any]:
        best_score = float('-inf')
        best_params = {}
        for params in self.enumerate_combinations():
            model = build_model(model_type, X.shape[1],
                                y.shape[1] if len(y.shape) > 1 else 1)
            if hasattr(model, 'train'):
                import inspect as _ins
                sig = _ins.signature(model.train)
                valid = {k: v for k, v in params.items() if k in sig.parameters}
                model.train(X, y, epochs=5, verbose=False, **valid)
            score = validate_model(model, X, y, metric=metric)
            if score > best_score:
                best_score = score
                best_params = params
        return {'best_score': best_score, 'best_params': best_params}


