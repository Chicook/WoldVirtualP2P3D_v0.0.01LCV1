import numpy as np


class GaussianSplatNeuron:
    """Gaussian splats preview: almacena N gaussianas con media, cov y opacidad.
    """

    def __init__(self, num_gaussians: int = 8):
        rng = np.random.default_rng(23)
        self.means = (rng.uniform(-1, 1, (num_gaussians, 3))).astype(np.float32)
        self.scales = (rng.uniform(0.05, 0.25, (num_gaussians, 3))).astype(np.float32)
        self.opacity = (rng.uniform(0.2, 0.9, (num_gaussians, 1))).astype(np.float32)
        self.colors = (rng.uniform(0, 1, (num_gaussians, 3))).astype(np.float32)

    def forward(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(points).astype(np.float32)
        # densidad aproximada como suma de gauss separables
        diffs = points[:, None, :] - self.means[None, :, :]
        exps = np.exp(-0.5 * ((diffs / (self.scales[None, :, :] + 1e-6)) ** 2).sum(axis=2))
        dens = (exps * self.opacity.squeeze(-1)[None, :]).sum(axis=1)
        return dens
