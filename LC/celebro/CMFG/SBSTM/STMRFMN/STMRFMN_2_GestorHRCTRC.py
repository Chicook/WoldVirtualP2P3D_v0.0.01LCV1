class GestorHRCTRC:
    def __init__(self):
        self.reglas: List[Dict[str, Any]] = []
        self._activa: bool = False

    def activar(self) -> None:
        self._activa = True
        logger.info("[HRCTRC] Constructor activado.")

    def desactivar(self) -> None:
        self._activa = False

    def esta_activa(self) -> bool:
        return self._activa

    def agregar_regla(self, condicion: str, accion: str) -> None:
        self.reglas.append({"condicion": condicion, "accion": accion})

    def listar_reglas(self) -> List[Dict[str, str]]:
        return list(self.reglas)

    def eliminar_regla(self, condicion: str) -> bool:
        for i, r in enumerate(self.reglas):
            if r["condicion"] == condicion:
                del self.reglas[i]
                return True
        return False

    def evaluar(self, prompt: str) -> Optional[str]:
        for regla in self.reglas:
            if regla["condicion"] in prompt:
                return regla["accion"]
        return None


# ═══════════════════════════════════════════════════════════════════
# CLASS 5: Cliente IA Free - interfaz al modelo de IA local/remoto
# ═══════════════════════════════════════════════════════════════════
