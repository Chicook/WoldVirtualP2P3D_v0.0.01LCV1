"""Hipotalamo/homeostasis.py - Regulacion energetica y ritmos de LucIA."""
import math


class Homeostasis:
    """Controla energia, fatiga acumulada y necesidad de descanso.
    Todo en RAM; a_pesos() lo vuelca el ModuloCerebral base."""

    def __init__(self):
        self.energia = 1.0
        self.fatiga = 0.0
        self.ciclos = 0
        self.historial = []

    def turno(self, delta: float, emocion: float = 0.0) -> dict:
        esfuerzo = min(0.25, max(0.0, float(delta)) / 20.0)
        extra = 0.05 * min(1.0, abs(float(emocion)))
        self.energia = round(max(0.0, min(1.0, self.energia - esfuerzo - extra)), 4)
        self.fatiga = round(min(1.0, self.fatiga + esfuerzo * 0.5), 4)
        self.ciclos += 1
        self.historial.append((self.ciclos, self.energia, self.fatiga))
        if len(self.historial) > 200:
            self.historial.pop(0)
        return {"energia": self.energia, "fatiga": self.fatiga,
                "pide_descanso": self.energia < 0.25 or self.fatiga > 0.8}

    def dormir(self, segundos: int = 20) -> dict:
        recup = min(1.0, segundos / 60.0)
        self.energia = round(min(1.0, self.energia + recup), 4)
        self.fatiga = round(max(0.0, self.fatiga - recup), 4)
        return {"energia": self.energia, "fatiga": self.fatiga}

    def ritmo(self) -> dict:
        if len(self.historial) < 4:
            return {"tendencia": "estable", "pendiente": 0.0}
        xs = [h[0] for h in self.historial[-20:]]
        ys = [h[1] for h in self.historial[-20:]]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs) or 1.0
        pendiente = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
        tend = "cansandose" if pendiente < -0.001 else "recuperando" if pendiente > 0.001 else "estable"
        return {"tendencia": tend, "pendiente": round(pendiente, 5)}
