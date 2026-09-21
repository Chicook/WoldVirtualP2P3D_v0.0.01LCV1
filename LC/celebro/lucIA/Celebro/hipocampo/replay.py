"""hipocampo/replay.py - Estrategias de replay (sueno) del hipocampo."""
import math


class EstrategiasReplay:
    """Ordena pendientes para consolidar: por importancia, por novedad o
    mezclado (como el sueno biologico: primero lo intenso, luego repaso)."""

    MODOS = ("importancia", "novedad", "mezclado")

    def ordenar(self, pendientes: list, modo: str = "mezclado") -> list:
        pend = list(pendientes)
        if modo == "novedad":
            return sorted(pend, key=lambda t: t.get("delta", 0), reverse=True)
        if modo == "importancia":
            return sorted(pend, key=lambda t: t.get("importancia", 0), reverse=True)
        mitad = len(pend) // 2
        top = sorted(pend, key=lambda t: t.get("importancia", 0), reverse=True)[:mitad]
        resto = [t for t in pend if t not in top]
        nov = sorted(resto, key=lambda t: t.get("delta", 0), reverse=True)
        return top + nov

    def lote(self, pendientes: list, n: int = 5, modo: str = "mezclado") -> list:
        return self.ordenar(pendientes, modo)[:n]
