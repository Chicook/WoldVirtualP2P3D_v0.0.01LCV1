"""Ganglios_basales/habitos.py - Tabla de habitos por recompensa."""
import time


class TablaHabitos:
    """Refuerza conductas: lo premiado sube, lo castigado baja. Extincion lenta."""

    def __init__(self, extincion: float = 0.99):
        self.extincion = extincion
        self.tabla = {}

    def reforzar(self, conducta: str, signo: int) -> float:
        v = self.tabla.get(conducta, {"valor": 0.0, "n": 0})
        v["valor"] = round(max(-1.0, min(1.0, v["valor"] + 0.2 * signo)), 3)
        v["n"] += 1
        v["ultimo"] = round(time.time(), 0)
        self.tabla[conducta] = v
        return v["valor"]

    def latir(self) -> None:
        for v in self.tabla.values():
            v["valor"] = round(v["valor"] * self.extincion, 4)

    def mejores(self, n: int = 3) -> list:
        return sorted(((k, v["valor"]) for k, v in self.tabla.items()),
                      key=lambda kv: kv[1], reverse=True)[:n]

    def valor(self, conducta: str) -> float:
        return self.tabla.get(conducta, {}).get("valor", 0.0)
