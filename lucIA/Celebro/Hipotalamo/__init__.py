"""Hipotalamo - Homeostasis de LucIA: energia, sueno y ritmo (basico)."""
from lucIA.Celebro.bases_neuro import ModuloCerebral


class Hipotalamo(ModuloCerebral):
    nombre = "Hipotalamo"

    def __init__(self):
        super().__init__()
        self.energia = 1.0

    def latido(self, delta: float) -> float:
        """Cada turno gasta energia segun el esfuerzo (delta); el descanso la repone."""
        self.energia = round(max(0.0, min(1.0, self.energia - min(0.2, float(delta) / 25.0))), 3)
        self._notar(f"energia={self.energia}")
        return self.energia

    def reponer(self) -> float:
        self.energia = 1.0
        self._notar("reposo completo")
        return self.energia

    def resumen(self) -> str:
        return f"Hipotalamo: energia={self.energia} | {super().resumen()}"


_hipotalamo_global = None

def get_hipotalamo() -> Hipotalamo:
    global _hipotalamo_global
    if _hipotalamo_global is None:
        _hipotalamo_global = Hipotalamo()
    return _hipotalamo_global
