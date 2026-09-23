class BackpropagationOptimizerInternal:
    """Motor interno de Backpropagation con Momentum + Muon + SOAP (2026)."""

    def __init__(self, learning_rate: float, momentum: float,
                 nesterov: bool, weight_decay: float,
                 ns_steps: int = 5, use_soap: bool = True,
                 polyak_decay: float = 0.999, trust_clip: float = 10.0):
        self.lr, self.momentum   = learning_rate, momentum
        self.nesterov            = nesterov
        self.weight_decay        = weight_decay
        self.ns_steps            = ns_steps
        self.use_soap            = use_soap
        self.polyak_decay        = polyak_decay
        self.trust_clip          = trust_clip
        # Estado interno
        self._velocity: Optional[np.ndarray]  = None
        self._L: Optional[np.ndarray]          = None
        self._R: Optional[np.ndarray]          = None
        self._polyak: Optional[np.ndarray]     = None
        self._G_sum: Optional[np.ndarray]      = None
        self._G_sq:  Optional[np.ndarray]      = None
        self.step_count = 0
        self.bp_score   = 0.0
        self.muon_orthogonality = 0.0
        self.soap_norm  = 0.0
        self.gsnr_score = 0.0
        self.trust_ratio_mean = 0.0

    def _init_state(self, G: np.ndarray) -> None:
        m, n = G.shape
        self._velocity = np.zeros_like(G)
        self._polyak   = np.zeros_like(G)
        self._G_sum    = np.zeros_like(G)
        self._G_sq     = np.zeros_like(G)
        self._L = np.eye(m) * 1e-4
        self._R = np.eye(n) * 1e-4

    def step(self, G: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Paso de optimizacion Backpropagation 2026.

        Si G es None, genera un gradiente sintetico para demostracion.
        Devuelve metricas del paso.
        """
        self.step_count += 1
        t = self.step_count

        # Gradiente sintetico si no se proporciona
        if G is None:
            size = 8
            G = np.random.randn(size, size) * 0.1

        if self._velocity is None:
            self._init_state(G)

        # --- Decaimiento de pesos (L2) ---
        G = G + self.weight_decay * np.random.randn(*G.shape) * 0.01

        # --- Momentum / Nesterov ---
        if self.nesterov:
            v_look = self.momentum * self._velocity + G
            self._velocity[:] = v_look
            G_eff = G + self.momentum * v_look
        else:
            self._velocity[:] = self.momentum * self._velocity + G
            G_eff = self._velocity.copy()

        # --- Muon Newton-Schulz 5 ---
        G_orth = MathematicalPrecision2026.newton_schulz5(G_eff, self.ns_steps)
        orth = float(np.trace(G_orth.T @ G_eff) /
                     (np.linalg.norm(G_eff, "fro") * np.linalg.norm(G_orth, "fro")
                      + MathematicalPrecision2026.EPS))
        self.muon_orthogonality = max(0.0, min(1.0, orth))

        # --- SOAP preconditioning ---
        if self.use_soap:
            G_pre = MathematicalPrecision2026.soap_precondition(
                G_orth, self._L, self._R)
            self.soap_norm = float(np.linalg.norm(G_pre, "fro"))
        else:
            G_pre = G_orth

        # --- Trust-ratio clip ---
        dummy_param = np.ones_like(G_pre)
        G_clip = MathematicalPrecision2026.trust_ratio_clip(
            G_pre, dummy_param, self.trust_clip)
        self.trust_ratio_mean = float(
            np.linalg.norm(G_clip) / (np.linalg.norm(G_pre) + MathematicalPrecision2026.EPS))

        # --- Update & Polyak ---
        update = self.lr * G_clip
        self._polyak[:] = MathematicalPrecision2026.polyak_average(
            self._polyak, update, self.polyak_decay)

        # --- GSNR ---
        self._G_sum += G
        self._G_sq  += G ** 2
        self.gsnr_score = MathematicalPrecision2026.gsnr(
            self._G_sum, self._G_sq, t)

        # --- BP score sintetico ---
        self.bp_score = float(np.clip(
            0.5 * self.muon_orthogonality + 0.3 * min(self.gsnr_score, 1.0)
            + 0.2 * self.trust_ratio_mean, 0.0, 1.0))

        return {
            "bp_score":          self.bp_score,
            "muon_orthogonality": self.muon_orthogonality,
            "soap_norm":         self.soap_norm,
            "gsnr":              self.gsnr_score,
            "trust_ratio":       self.trust_ratio_mean,
        }


# ===========================================================================
# 3. OPTIMIZADOR PRINCIPAL
# ===========================================================================
