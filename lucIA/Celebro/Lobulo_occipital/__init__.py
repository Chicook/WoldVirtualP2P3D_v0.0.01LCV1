"""Lobulo_occipital - Vision de LucIA (stub basico: reservado)."""
from lucIA.Celebro.bases_neuro import ModuloCerebral


class LobuloOccipital(ModuloCerebral):
    nombre = "Occipital"

    def ver(self, imagen=None) -> str:
        """Sin vision aun: documenta el intento para futuro cableado."""
        self._notar("intento visual (sin sensor)")
        return "sin vision"

    def resumen(self) -> str:
        return f"Occipital: stub | {super().resumen()}"


_occipital_global = None

def get_occipital() -> LobuloOccipital:
    global _occipital_global
    if _occipital_global is None:
        _occipital_global = LobuloOccipital()
    return _occipital_global
