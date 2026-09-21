"""Cuerpo_calloso - Puente entre hemisferios local y cloud (basico)."""
from lucIA.Celebro.bases_neuro import ModuloCerebral


class CuerpoCalloso(ModuloCerebral):
    nombre = "CuerpoCalloso"

    def __init__(self):
        super().__init__()
        self.ultima_integracion = ""

    def integrar(self, via_local: str, via_cloud: str) -> str:
        """Fusiona las dos vias en una sola linea (lo esencial de ambas)."""
        self.ultima_integracion = f"[L] {(via_local or '')[:120]} | [C] {(via_cloud or '')[:120]}"
        self._notar(f"integracion chars={len(self.ultima_integracion)}")
        return self.ultima_integracion

    def resumen(self) -> str:
        return f"CuerpoCalloso: {'integrado' if self.ultima_integracion else 'sin integrar'} | {super().resumen()}"


_calloso_global = None

def get_calloso() -> CuerpoCalloso:
    global _calloso_global
    if _calloso_global is None:
        _calloso_global = CuerpoCalloso()
    return _calloso_global
