class ConfiguracionLucia:
    def __init__(self, ruta_base: str = ".", sesion_id: Optional[str] = None):
        self.ruta_base = Path(ruta_base)
        self.sesion_id = sesion_id or hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]
        self.max_turnos: int = 1000
        self.max_neuronas: int = 5000
        self.ruta_pesos: Path = self.ruta_base / "pesos"
        self.ruta_md: Path = self.ruta_base / "memoria.md"
        self.ruta_ipfs: Path = self.ruta_base / "ipfs"
        self.ia_local_lista: bool = True
        self.debug: bool = False

    def validar_rutas(self) -> bool:
        for ruta in (self.ruta_pesos, self.ruta_ipfs):
            ruta.mkdir(parents=True, exist_ok=True)
        return True

    def como_diccionario(self) -> Dict[str, Any]:
        return {"sesion_id": self.sesion_id, "max_turnos": self.max_turnos,
                "max_neuronas": self.max_neuronas, "ia_local_lista": self.ia_local_lista, "debug": self.debug}

    def __repr__(self) -> str:
        return f"ConfiguracionLucia(sesion={self.sesion_id}, rutas={self.ruta_base})"

    def reset(self) -> None:
        """Resetea los valores configurables a sus defaults."""
        self.max_turnos = 1000
        self.max_neuronas = 5000
        self.ia_local_lista = True
        self.debug = False


# ═══════════════════════════════════════════════════════════════════
# CLASS 2: Servidor BKS - cadena de bloques ligera para trazabilidad
# ═══════════════════════════════════════════════════════════════════
