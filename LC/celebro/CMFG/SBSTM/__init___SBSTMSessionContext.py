class SBSTMSessionContext:
    """
    Context Manager profesional para encapsular ejecuciones por lotes o pruebas
    de red dentro del ciclo de vida controlado de una sesion neuronal P2P.
    """

    def __init__(self, modelo: Optional[str] = None, puerto_blockchain: int = 8545) -> None:
        self.modelo = modelo
        self.puerto_blockchain = puerto_blockchain
        self.sesion: Optional[Any] = None

    def __enter__(self) -> Any:
        self.sesion = crear_sesion_neuronal(
            modelo=self.modelo,
            puerto_blockchain=self.puerto_blockchain,
        )
        return self.sesion

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if self.sesion is not None and hasattr(self.sesion, "cerrar"):
            try:
                self.sesion.cerrar()
            except Exception as e:
                logger.warning("Error durante cierre automatico de sesion SBSTM: %s", e)
        with _LOCK:
            _SBSTM_STATE["session_active"] = False
            _SBSTM_STATE["active_instance"] = None
        return False


# ─── SERVICIO DE DISCOVERY DE MODELOS OLLAMA LOCALES ────────────────────────
