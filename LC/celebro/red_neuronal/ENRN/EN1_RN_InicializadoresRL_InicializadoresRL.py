class InicializadoresRL:
    """Coleccion de inicializadores usados en RL."""

    @staticmethod
    def _rng(seed: int = 42) -> np.random.Generator:
        return np.random.default_rng(seed)

    @classmethod
    def he(cls, shape, fan_in=1, seed=42):
        return cls._rng(seed).normal(
            0.0, math.sqrt(2.0 / max(1, fan_in)), shape,
        ).astype(LUCIA_RL_CONFIG['precision'])

    @classmethod
    def xavier(cls, shape, fan_in=1, fan_out=1, seed=42):
        lim = math.sqrt(6.0 / max(1, fan_in + fan_out))
        return cls._rng(seed).uniform(-lim, lim, shape).astype(
            LUCIA_RL_CONFIG['precision']
        )

    @classmethod
    def lecun(cls, shape, fan_in=1, seed=42):
        """Inicializacion LeCun (recomendada para SELU)."""
        lim = math.sqrt(3.0 / max(1, fan_in))
        return cls._rng(seed).uniform(-lim, lim, shape).astype(
            LUCIA_RL_CONFIG['precision']
        )

    @classmethod
    def ortogonal(cls, shape, seed=42):
        """Inicializacion ortogonal (preserva norma)."""
        flat = cls._rng(seed).standard_normal((shape[0], shape[-1]))
        q, r = np.linalg.qr(flat)
        q = q.astype(LUCIA_RL_CONFIG['precision'])
        if len(shape) > 2:
            q = q.reshape(shape)
        return q

    @classmethod
    def espectral(cls, shape, seed=42):
        """Inicializacion por descomposicion espectral."""
        mat = cls._rng(seed).standard_normal(shape).astype(LUCIA_RL_CONFIG['precision'])
        try:
            u, _, vh = np.linalg.svd(mat, full_matrices=False)
            return u @ vh
        except np.linalg.LinAlgError:
            return cls.uniforme(shape, seed=seed)

    @classmethod
    def uniforme(cls, shape, scale=0.1, seed=42):
        """Inicializacion uniforme simple."""
        return cls._rng(seed).uniform(-scale, scale, shape).astype(
            LUCIA_RL_CONFIG['precision']
        )


