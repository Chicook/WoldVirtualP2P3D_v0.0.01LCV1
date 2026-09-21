"""Ganglios_basales - Recompensa y habitos de LucIA."""
from lucIA.Celebro.bases_neuro import ModuloCerebral


class GangliosBasales(ModuloCerebral):
    nombre = "Ganglios"

    def __init__(self):
        super().__init__()
        self.balance = 0
        self.habitos = {}

    def recompensar(self, signo: int, conversor, motivo: str = "") -> int:
        signo = 1 if signo > 0 else -1
        self.balance += signo
        self._notar(f"{'+' if signo > 0 else '-'}1 {motivo[:60]}")
        try:
            v = conversor.encoder.encode(f"[recompensa {'positiva' if signo > 0 else 'negativa'}] {motivo[:200]}")
            act, _ = conversor._propagar_todas_las_neuronas(v)
            conversor._actualizar_pesos_en_todas_las_neuronas(
                v, act, factor=0.8 * signo, fase="recompensa")
        except Exception:
            pass
        return self.balance

    def habito(self, clave: str) -> int:
        self.habitos[clave] = self.habitos.get(clave, 0) + 1
        return self.habitos[clave]

    def resumen(self) -> str:
        return f"Ganglios: balance={self.balance} habitos={len(self.habitos)} | {super().resumen()}"


_ganglios_global = None

def get_ganglios() -> GangliosBasales:
    global _ganglios_global
    if _ganglios_global is None:
        _ganglios_global = GangliosBasales()
    return _ganglios_global
