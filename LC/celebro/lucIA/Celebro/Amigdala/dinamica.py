"""Amigdala/dinamica.py - Dinamica afectiva: la perilla del peso."""
import math


class DinamicaAfectiva:
    """El estado afectivo decae con el tiempo (vuelve a la calma) y salta con
    eventos intensos. Su salida pondera cuanto peso gana cada recuerdo."""

    def __init__(self, decaimiento: float = 0.9):
        self.nivel = 0.0
        self.signo = 1.0
        self.decaimiento = decaimiento
        self.picos = 0

    def evento(self, emocion: float, delta: float = 0.0) -> float:
        impulso = min(1.0, abs(float(emocion)) * 0.7 + min(1.0, max(0.0, float(delta)) / 5.0) * 0.3)
        if impulso > self.nivel:
            self.nivel = round(impulso, 4)
            self.signo = 1.0 if emocion >= 0 else -1.0
            if impulso > 0.6:
                self.picos += 1
        return self.nivel * self.signo

    def latir(self) -> float:
        """Decaimiento por turno sin estimulo: vuelve a la calma."""
        self.nivel = round(self.nivel * self.decaimiento, 4)
        return self.nivel * self.signo

    def factor_peso(self) -> float:
        """0.5 (calma) .. 1.5 (pico): controla el peso de la consolidacion."""
        return round(0.5 + abs(self.nivel), 3)

    def estado(self) -> dict:
        return {"nivel": self.nivel, "signo": self.signo,
                "factor_peso": self.factor_peso(), "picos": self.picos}
