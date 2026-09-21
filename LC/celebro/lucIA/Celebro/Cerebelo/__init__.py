"""Cerebelo - Coordinacion fina, timing y marcas para lip-sync."""
import time
from lucIA.Celebro.bases_neuro import ModuloCerebral


class Cerebelo(ModuloCerebral):
    nombre = "Cerebelo"

    def __init__(self):
        super().__init__()
        self.marcas = {}

    def marcar(self, fase: str) -> None:
        self.marcas[fase] = time.perf_counter()
        self._notar(fase)

    def ritmo(self) -> dict:
        fs = list(self.marcas)
        lat = {}
        for i in range(1, len(fs)):
            lat[f"{fs[i-1]}->{fs[i]}"] = round(self.marcas[fs[i]] - self.marcas[fs[i-1]], 3)
        return {"fases": fs, "latencias_s": lat}

    def marcas_visemas(self, texto: str) -> list:
        """Stub lip-sync Godot (puertos 9876/9877): reparte el texto en
        ventanas de 120ms para futuro mapeo a visemas."""
        n = max(1, len(texto or "") // 12)
        return [{"t_ms": i * 120, "ventana": (texto or "")[i*12:(i+1)*12]} for i in range(min(n, 40))]

    def resumen(self) -> str:
        return f"Cerebelo: fases={len(self.marcas)} | {super().resumen()}"


_cerebelo_global = None

def get_cerebelo() -> Cerebelo:
    global _cerebelo_global
    if _cerebelo_global is None:
        _cerebelo_global = Cerebelo()
    return _cerebelo_global
