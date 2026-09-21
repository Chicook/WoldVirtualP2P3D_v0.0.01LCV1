import numpy as np


class HashGridNGPNeuron:
    """Instant-NGP like preview: hash embedding + MLP pequeño.
    Para preview solo definimos 'weights' de embedding y capas.
    """

    def __init__(self, levels: int = 4, table_size: int = 64, feat_dim: int = 2, hidden: int = 16):
        rng = np.random.default_rng(13)
        self.levels = levels
        self.tables = [
            (rng.standard_normal((table_size, feat_dim)) * 0.1).astype(np.float32)
            for _ in range(levels)
        ]
        self.w_mlp = (rng.standard_normal((levels * feat_dim, hidden)) * (1.0 / (levels * feat_dim))).astype(np.float32)
        self.b_mlp = (rng.standard_normal((hidden,)) * 0.1).astype(np.float32)
        self.w_out = (rng.standard_normal((hidden, 1)) * (1.0 / hidden)).astype(np.float32)

    def _embed(self, coords: np.ndarray) -> np.ndarray:
        coords = np.clip(coords, 0, 1)
        idx = np.minimum((coords * (len(self.tables[0]) - 1)).astype(np.int32), len(self.tables[0]) - 1)
        feats = [tbl[idx[:, 0] % len(tbl)] for tbl in self.tables]
        return np.concatenate(feats, axis=1)

    def forward(self, coords: np.ndarray) -> np.ndarray:
        coords = np.atleast_2d(coords).astype(np.float32)
        emb = self._embed(coords[:, :1])  # usar primera coord para index simple
        h = np.maximum(0, emb @ self.w_mlp + self.b_mlp)
        return (h @ self.w_out).squeeze(-1)
