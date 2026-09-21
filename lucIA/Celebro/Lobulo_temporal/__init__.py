"""Lobulo_temporal - Lenguaje y oido de LucIA (basico)."""
from lucIA.Celebro.bases_neuro import ModuloCerebral


class LobuloTemporal(ModuloCerebral):
    nombre = "Temporal"

    def __init__(self):
        super().__init__()
        self.frases_oidas = 0

    def oir(self, texto: str) -> int:
        """Registra una frase oida/hablada (memoria auditiva basica)."""
        if texto and len(texto.strip()) >= 3:
            self.frases_oidas += 1
            self._notar(f"oido #{self.frases_oidas}: {texto.strip()[:60]}")
        return self.frases_oidas

    def resumen(self) -> str:
        return f"Temporal: frases={self.frases_oidas} | {super().resumen()}"


_temporal_global = None

def get_temporal() -> LobuloTemporal:
    global _temporal_global
    if _temporal_global is None:
        _temporal_global = LobuloTemporal()
    return _temporal_global
