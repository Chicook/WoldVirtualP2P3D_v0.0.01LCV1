"""Corteza_sensorial - Vias de entrada de sentidos de LucIA."""
import re
from lucIA.Celebro.bases_neuro import ModuloCerebral


class CortezaSensorial(ModuloCerebral):
    nombre = "CortezaSensorial"
    VIAS = ("texto",)

    def __init__(self):
        super().__init__()
        self.vias = {"texto": True}

    def normalizar(self, texto: str) -> str:
        t = re.sub(r"\s+", " ", (texto or "")).strip()
        self._notar(f"via=texto chars={len(t)}")
        return t

    def registrar_via(self, nombre: str) -> bool:
        """Stub para futuras vias (vision, audio). Hoy solo documenta."""
        if nombre in self.VIAS or nombre in ("vision", "audio"):
            self.vias[nombre] = (nombre == "texto")
            self._notar(f"via registrada: {nombre}")
            return True
        return False

    def resumen(self) -> str:
        return f"Sensorial: vias={sorted(self.vias)} | {super().resumen()}"


_sensorial_global = None

def get_sensorial() -> CortezaSensorial:
    global _sensorial_global
    if _sensorial_global is None:
        _sensorial_global = CortezaSensorial()
    return _sensorial_global
