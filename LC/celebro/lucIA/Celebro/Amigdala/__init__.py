"""Amigdala - Etiqueta emocional y alerta de LucIA."""
from lucIA.Celebro.bases_neuro import ModuloCerebral


class Amigdala(ModuloCerebral):
    nombre = "Amigdala"

    def __init__(self):
        super().__init__()
        self.activacion = 0.0
        self.etiqueta = "calma"

    def evaluar(self, emocion: float, delta: float = 0.0) -> dict:
        e = abs(float(emocion))
        d = min(1.0, float(delta) / 5.0) if delta > 0 else 0.0
        self.activacion = round(min(1.0, 0.7 * e + 0.3 * d), 3)
        if self.activacion > 0.6:
            self.etiqueta = "alerta"
        elif self.activacion > 0.3:
            self.etiqueta = "entusiasmo" if emocion > 0 else "inquietud"
        else:
            self.etiqueta = "calma"
        self._notar(f"{self.etiqueta} act={self.activacion}")
        return {"activacion": self.activacion, "etiqueta": self.etiqueta,
                "necesita_descanso": self.activacion > 0.8}

    def ponderar(self, emocion: float) -> float:
        """Perilla unica del peso emocional: el resto de modulos deben usar
        este valor (activacion con signo) en vez de la emocion cruda."""
        return round(float(self.activacion) * (1.0 if emocion >= 0 else -1.0), 3)

    def resumen(self) -> str:
        return f"Amigdala: {self.etiqueta} ({self.activacion}) | {super().resumen()}"


_amigdala_global = None

def get_amigdala() -> Amigdala:
    global _amigdala_global
    if _amigdala_global is None:
        _amigdala_global = Amigdala()
    return _amigdala_global
