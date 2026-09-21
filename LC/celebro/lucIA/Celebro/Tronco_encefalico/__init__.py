"""Tronco_encefalico - Funciones vitales: vigilancia y descanso."""
import time
from lucIA.Celebro.bases_neuro import ModuloCerebral


class TroncoEncefalico(ModuloCerebral):
    nombre = "Tronco"

    def __init__(self):
        super().__init__()
        self.descansos = 0

    def vigilar(self) -> dict:
        try:
            from lucIA.session_manager import verificar_limite_espacio
            info = verificar_limite_espacio()
        except Exception:
            info = {"tamano_mb": -1.0, "limite_mb": 125.0, "valido": True}
        self._notar(f"vigilia mb={info.get('tamano_mb')}")
        return info

    def descanso(self, segundos: int = 20) -> None:
        segundos = max(5, min(int(segundos), 120))
        self.descansos += 1
        self._notar(f"descanso {segundos}s")
        time.sleep(segundos)

    def resumen(self) -> str:
        return f"Tronco: descansos={self.descansos} | {super().resumen()}"


_tronco_global = None

def get_tronco() -> TroncoEncefalico:
    global _tronco_global
    if _tronco_global is None:
        _tronco_global = TroncoEncefalico()
    return _tronco_global
