"""Talamo - Puerta sensorial explicita de LucIA."""
import numpy as np
from lucIA.Celebro.bases_neuro import ModuloCerebral


class Talamo(ModuloCerebral):
    nombre = "Talamo"

    def __init__(self, alfa_ema: float = 0.85):
        super().__init__()
        self.alfa = alfa_ema
        self._ema = None
        self.bloqueados = 0

    def filtrar(self, vector):
        v = np.asarray(vector, dtype=np.float32)
        if self._ema is None:
            self._ema = v.copy()
        self._ema = self.alfa * v + (1.0 - self.alfa) * self._ema
        norma = float(np.linalg.norm(v - self._ema))
        if norma > 8.0:  # pico desproporcionado: ruido, se suaviza
            self.bloqueados += 1
            self._notar(f"pico filtrado norma={norma:.2f}")
            return np.clip(self._ema, -5.0, 5.0), f"filtrado (pico {norma:.2f})"
        return np.clip(self._ema, -5.0, 5.0), "pasa"

    def resumen(self) -> str:
        return f"Talamo: bloqueados={self.bloqueados} | {super().resumen()}"


_talamo_global = None

def get_talamo() -> Talamo:
    global _talamo_global
    if _talamo_global is None:
        _talamo_global = Talamo()
    return _talamo_global
