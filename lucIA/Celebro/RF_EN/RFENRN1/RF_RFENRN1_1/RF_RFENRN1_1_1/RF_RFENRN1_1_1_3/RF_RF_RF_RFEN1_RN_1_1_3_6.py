import numpy as np


class ProceduralNoiseNeuron:
    """Procedural noise preview: FBM con hash simple en numpy.
    Genera valores de ruido para coordenadas 2D/3D.
    """

    def __init__(self, octaves: int = 4):
        self.octaves = octaves
        rng = np.random.default_rng(77)
        self.grad = (rng.uniform(-1, 1, (256, 3))).astype(np.float32)
        self.perm = rng.integers(0, 256, size=(512,), endpoint=False).astype(np.int32)
        self.perm[:256] = np.arange(256)
        rng.shuffle(self.perm[:256])
        self.perm[256:] = self.perm[:256]

    def _fade(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _lerp(self, a, b, t):
        return a + t * (b - a)

    def _grad(self, hashv, x, y, z):
        g = self.grad[hashv % 256]
        return g[0] * x + g[1] * y + g[2] * z

    def noise3d(self, p: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(p).astype(np.float32)
        if p.shape[1] == 2:
            p = np.concatenate([p, np.zeros((p.shape[0], 1), dtype=p.dtype)], axis=1)
        xi = np.floor(p).astype(np.int32) & 255
        xf = p - np.floor(p)
        u, v, w = self._fade(xf[:, 0]), self._fade(xf[:, 1]), self._fade(xf[:, 2])
        aaa = self.perm[self.perm[self.perm[xi[:, 0]] + xi[:, 1]] + xi[:, 2]]
        aba = self.perm[self.perm[self.perm[xi[:, 0]] + xi[:, 1] + 1] + xi[:, 2]]
        aab = self.perm[self.perm[self.perm[xi[:, 0]] + xi[:, 1]] + xi[:, 2] + 1]
        abb = self.perm[self.perm[self.perm[xi[:, 0]] + xi[:, 1] + 1] + xi[:, 2] + 1]
        baa = self.perm[self.perm[self.perm[xi[:, 0] + 1] + xi[:, 1]] + xi[:, 2]]
        bba = self.perm[self.perm[self.perm[xi[:, 0] + 1] + xi[:, 1] + 1] + xi[:, 2]]
        bab = self.perm[self.perm[self.perm[xi[:, 0] + 1] + xi[:, 1]] + xi[:, 2] + 1]
        bbb = self.perm[self.perm[self.perm[xi[:, 0] + 1] + xi[:, 1] + 1] + xi[:, 2] + 1]
        x, y, z = xf[:, 0], xf[:, 1], xf[:, 2]
        x1 = self._lerp(self._grad(aaa, x, y, z), self._grad(baa, x - 1, y, z), u)
        x2 = self._lerp(self._grad(aba, x, y - 1, z), self._grad(bba, x - 1, y - 1, z), u)
        y1 = self._lerp(x1, x2, v)
        x1 = self._lerp(self._grad(aab, x, y, z - 1), self._grad(bab, x - 1, y, z - 1), u)
        x2 = self._lerp(self._grad(abb, x, y - 1, z - 1), self._grad(bbb, x - 1, y - 1, z - 1), u)
        y2 = self._lerp(x1, x2, v)
        return self._lerp(y1, y2, w)

    def forward(self, coords: np.ndarray) -> np.ndarray:
        val = 0.0
        amp = 1.0
        freq = 1.0
        coords = np.atleast_2d(coords).astype(np.float32)
        for _ in range(self.octaves):
            val = val + amp * self.noise3d(coords * freq)
            amp *= 0.5
            freq *= 2.0
        return val.astype(np.float32)
