"""Glia - Limpieza y verificacion de cierre limpio."""
from pathlib import Path
from lucIA.Celebro.bases_neuro import ModuloCerebral

_LUCIA_DIR = Path(__file__).parent.parent.resolve()


class Glia(ModuloCerebral):
    nombre = "Glia"

    def verificar(self) -> dict:
        """Lista restos de ejecucion: 0 .npz, 0 JSON de memoria, cache vacia."""
        restos = []
        for patron in ("*.npz", "memoria_conversacion.json", "pesos_*.npz"):
            for p in _LUCIA_DIR.rglob(patron):
                if "venv" not in str(p):
                    restos.append(str(p.relative_to(_LUCIA_DIR)))
        cache = _LUCIA_DIR / "Celebro" / "cache"
        if cache.exists():
            for item in cache.iterdir():
                restos.append(f"Celebro/cache/{item.name}")
        self._notar(f"verificacion restos={len(restos)}")
        return {"limpio": not restos, "restos": sorted(set(restos))}

    def resumen(self) -> str:
        return f"Glia: verificaciones={len(self.eventos)} | {super().resumen()}"


_glia_global = None

def get_glia() -> Glia:
    global _glia_global
    if _glia_global is None:
        _glia_global = Glia()
    return _glia_global
