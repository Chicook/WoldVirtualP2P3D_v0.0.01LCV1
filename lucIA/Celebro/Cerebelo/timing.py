"""Cerebelo/timing.py - Cronometro de fases y ritmo de sesion."""
import time


class CronometroFases:
    """Mide latencia por fase y calcula el ritmo (turnos/minuto)."""

    def __init__(self):
        self.marcas = {}
        self.ritmos = []

    def marcar(self, fase: str) -> None:
        self.marcas[fase] = time.perf_counter()

    def tramo(self, f1: str, f2: str) -> float:
        if f1 in self.marcas and f2 in self.marcas:
            return round(self.marcas[f2] - self.marcas[f1], 3)
        return -1.0

    def turno_completo(self, segundos: float) -> None:
        self.ritmos.append(float(segundos))
        if len(self.ritmos) > 30:
            self.ritmos.pop(0)

    def ritmo(self) -> dict:
        if not self.ritmos:
            return {"media_s": 0.0, "ritmo_min": 0.0}
        media = sum(self.ritmos) / len(self.ritmos)
        return {"media_s": round(media, 2),
                "ritmo_min": round(60.0 / max(0.1, media), 1)}
