"""Corteza_prefrontal - Objetivos multi-turno de LucIA."""
import re
from lucIA.Celebro.bases_neuro import ModuloCerebral


class CortezaPrefrontal(ModuloCerebral):
    nombre = "Prefrontal"

    def __init__(self):
        super().__init__()
        self.objetivo = ""

    def fijar_objetivo(self, texto: str) -> str:
        t = (texto or "").strip()[:200]
        if re.search(r"(?i)(plan|objetivo|meta|quiero|necesito|ayudame|elabora|crea|disena)", t):
            self.objetivo = t
            self._notar(f"objetivo: {t[:80]}")
        return self.objetivo

    def avance(self, respuesta: str) -> float:
        if not self.objetivo or not respuesta:
            return 0.0
        claves = [w.lower() for w in re.findall(r"[a-zA-Záéíóúñ]{4,}", self.objetivo)][:8]
        resp = respuesta.lower()
        hit = sum(1 for w in claves if w in resp)
        return round(hit / max(1, len(claves)), 2)

    def resumen(self) -> str:
        return f"Prefrontal: objetivo={'si' if self.objetivo else 'no'} | {super().resumen()}"


_prefrontal_global = None

def get_prefrontal() -> CortezaPrefrontal:
    global _prefrontal_global
    if _prefrontal_global is None:
        _prefrontal_global = CortezaPrefrontal()
    return _prefrontal_global
