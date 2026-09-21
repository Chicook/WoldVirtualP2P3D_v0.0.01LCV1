"""Glia/limpieza.py - Barredora de restos de ejecucion."""
from pathlib import Path

_BASE = Path(__file__).parent.parent.resolve()
PATRONES = ("*.npz", "memoria_conversacion.json")


class Barredora:
    """Detecta (no borra: el borrado lo hace session_manager) restos
    generados en tiempo de ejecucion fuera de venv/."""

    def escanear(self) -> dict:
        restos = []
        for patron in PATRONES:
            for p in _BASE.rglob(patron):
                if "venv" not in str(p):
                    restos.append(str(p.relative_to(_BASE)))
        cache = _BASE / "Celebro" / "cache"
        if cache.exists():
            for item in cache.iterdir():
                restos.append(f"Celebro/cache/{item.name}")
        restos = sorted(set(restos))
        return {"limpio": not restos, "restos": restos, "n": len(restos)}
