"""Lobulo_parietal - Integracion sensorial: quien/donde/cuando (basico)."""
import time
from lucIA.Celebro.bases_neuro import ModuloCerebral


class LobuloParietal(ModuloCerebral):
    nombre = "Parietal"

    def __init__(self):
        super().__init__()
        self.contexto = {}

    def situar(self, pregunta: str, modelo: str = "", turno: int = 0) -> dict:
        """Fusiona el contexto del turno en un solo mapa."""
        self.contexto = {"cuando": round(time.time(), 0), "modelo": modelo,
                         "turno": turno, "chars": len(pregunta or "")}
        self._notar(f"situado turno={turno} modelo={modelo}")
        return dict(self.contexto)

    def resumen(self) -> str:
        return f"Parietal: turno={self.contexto.get('turno', '-')} | {super().resumen()}"


_parietal_global = None

def get_parietal() -> LobuloParietal:
    global _parietal_global
    if _parietal_global is None:
        _parietal_global = LobuloParietal()
    return _parietal_global
